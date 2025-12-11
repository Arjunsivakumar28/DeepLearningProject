import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ..layers import ImageInputHead


# Utility Functions
class PositionalEncoding(layers.Layer):
    """The positional encoding funcion for learnable position weights."""
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        # D_model
        self.d_model = d_model

    def build(self, input_shape):
        # input_shape: [Batch, N, D_model]
        seq_len = input_shape[1]
        self.pos_emb_vec = self.add_weight(
            "pos_emb",
            shape=(1, seq_len, self.d_model),
            initializer="random_normal",
        )

    def call(self, x):
        # x: same shape as input_shape from build
        return x + self.pos_emb_vec

class ModalityEmbedding(layers.Layer):
    """The modality encoding funcion for learnable modality weights."""
    def __init__(self, d_model, num_modalities=4, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_modalities = num_modalities # 2 modalities

    def build(self, input_shape):
        self.mod_emb_table = self.add_weight(
            "mod_emb_table",
            shape=(self.num_modalities, self.d_model),
            initializer="random_normal",
        )

    def call(self, x, modality_id):
        # x: [Batch, N, D_model]; 
        # modality_id: scalar like 0,1,2, etc.
        # Always treat modality_id as scalar
        mod_emb = tf.gather(self.mod_emb_table, modality_id) # [D_model]
        mod_emb = tf.reshape(mod_emb, (1, 1, self.d_model)) # [1,1,D_model]
        mod_emb = tf.broadcast_to(mod_emb, tf.shape(x)) # [B,N,D]
        return x + mod_emb


class FeedForward(layers.Layer):
    """Transformer MLP block."""
    def __init__(self, d_model, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(d_ff, activation="gelu")
        self.dense2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=None):
        residual = x
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        x = self.norm(residual + x)
        return x


class TransformerSelfAttnBlock(layers.Layer):
    """Self attention plus FFN (For latent tower)"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dropout = layers.Dropout(dropout)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.ff = FeedForward(d_model, d_ff, dropout) # maybe skip

        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )
        self.linear = layers.Dense(d_model)

    def call(self, x, training=None):
        
        h = self.norm(x)
        h = self.attn(h, h, training=training)
        x = x + self.dropout(h, training=training)

        return x


class CrossAttentionBlock(layers.Layer):
    """Cross Attention: Queries (or latent block) attend to K and V blocks."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dropout = layers.Dropout(dropout)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.ff = FeedForward(d_model, d_ff, dropout) # maybe skip
        # self.linear = layers.Dense(d_model)  # simple linear

        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )

    def call(self, inputs, training=None, attention_mask=None):
        query, key_value = inputs
        residual = query
        query = self.norm(query)
        key_value = self.norm(key_value)
        x = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            attention_mask=attention_mask,
            training=training,
        )
        x = residual + self.dropout(x, training=training)
        return x


# Patchtst encoder
class PatchTSTEncoder(layers.Layer):
    """
    PatchTST-style encoder:
      Input:  [B, S, F]
      Output: [B, N_ts_tokens, D_ts]
    """
    def __init__(self, patch_len=4, stride=4, d_ts=128, **kwargs):
        super().__init__(**kwargs)
        self.patch_len = patch_len
        self.stride = stride
        self.d_ts = d_ts

    def build(self, input_shape):
        self.proj = layers.Dense(self.d_ts)

    def call(self, x):
        # x: [B, S, F]
        x = tf.transpose(x, perm=[0, 2, 1])  # [B, F, S]
        frames = tf.signal.frame(
            x,
            frame_length=self.patch_len,
            frame_step=self.stride,
            axis=-1,
        )  # [B, F, num_patches, patch_len]

        B = tf.shape(frames)[0]
        F = tf.shape(frames)[1]
        num_patches = tf.shape(frames)[2]
        frames = tf.reshape(frames, (B, F * num_patches, self.patch_len))  # [B, N_ts, patch_len]

        tokens = self.proj(frames)  # [B, N_ts, D_ts]
        return tokens

# MAE style masking and dropping
class MaskingAndModDrop(layers.Layer):
    """
    MultiMAE-style masking for both modalities + ModDrop.
    """
    def __init__(
        self,
        d_model,
        img_mask_ratio=0.3,
        ts_mask_ratio=0.3,
        p_moddrop=0.2,
        num_modalities=2, # 0=image, 1=time series
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.img_mask_ratio = img_mask_ratio
        self.ts_mask_ratio = ts_mask_ratio
        self.p_moddrop = p_moddrop
        self.num_modalities = num_modalities

    def build(self, input_shape):
        # patch-level mask tokens (per modality)
        # [[image, d_model], [ts, d_model]]
        self.patch_mask_tokens = self.add_weight(
            "patch_mask_tokens",
            shape=(self.num_modalities, self.d_model),
            initializer="random_normal",
        )
        # to drop the whole modality mask
        self.mod_mask_tokens = self.add_weight(
            "mod_mask_tokens",
            shape=(self.num_modalities, self.d_model),
            initializer="random_normal",
        )

    def _mask_tokens(self, tokens, mask_ratio, patch_mask_token, training):
        # tokens: [B, N, D]
        B = tf.shape(tokens)[0]
        N = tf.shape(tokens)[1]
        if not training or mask_ratio <= 0.0:
            mask = tf.zeros((B, N), dtype=tf.bool)
            return tokens, mask

        # random masking for patches
        rand = tf.random.uniform((B, N))
        mask = rand < mask_ratio  # [B, N] bool
        mask_f = tf.cast(mask, tokens.dtype)[..., None]  # [B, N, 1]

        # for 1 modality
        patch_mask_token = tf.reshape(patch_mask_token, (1, 1, self.d_model))
        patch_mask_token = tf.broadcast_to(patch_mask_token, tf.shape(tokens))
        masked_tokens = tokens * (1.0 - mask_f) + patch_mask_token * mask_f
        return masked_tokens, mask

    def _moddrop(self, tokens, mod_mask_token, training):
        # tokens: [B, N, D]
        B = tf.shape(tokens)[0]
        if not training or self.p_moddrop <= 0.0:
            dropped = tf.zeros((B,), dtype=tf.bool)
            return tokens, dropped

        rand = tf.random.uniform((B,))
        dropped = rand < self.p_moddrop  # [B]
        dropped_f = tf.cast(dropped, tokens.dtype)[:, None, None]  # [B, 1, 1]

        mod_mask_token = tf.reshape(mod_mask_token, (1, 1, self.d_model))
        mod_mask_token = tf.broadcast_to(mod_mask_token, tf.shape(tokens))
        tokens_out = tokens * (1.0 - dropped_f) + mod_mask_token * dropped_f
        return tokens_out, dropped

    def call(self, inputs, training=None):
        img_tokens, ts_tokens = inputs  # both [B, N, D_model]

        # 1) Patch-level masking
        img_patch_token = self.patch_mask_tokens[0]
        ts_patch_token = self.patch_mask_tokens[1]

        img_masked, img_mask = self._mask_tokens(
            img_tokens, self.img_mask_ratio, img_patch_token, training
        )
        ts_masked, ts_mask = self._mask_tokens(
            ts_tokens, self.ts_mask_ratio, ts_patch_token, training
        )

        # 2) Modality dropout (ModDrop)
        img_mod_token = self.mod_mask_tokens[0]
        ts_mod_token = self.mod_mask_tokens[1]

        img_moddropped, img_dropped = self._moddrop(img_masked, img_mod_token, training)
        ts_moddropped, ts_dropped = self._moddrop(ts_masked, ts_mod_token, training)

        # Effective masks for reconstruction: only count if modality is NOT dropped
        img_dropped_broadcast = tf.broadcast_to(
            tf.expand_dims(img_dropped, 1), tf.shape(img_mask)
        )
        ts_dropped_broadcast = tf.broadcast_to(
            tf.expand_dims(ts_dropped, 1), tf.shape(ts_mask)
        )

        img_eff_mask = tf.logical_and(img_mask, tf.logical_not(img_dropped_broadcast))
        ts_eff_mask = tf.logical_and(ts_mask, tf.logical_not(ts_dropped_broadcast))

        return img_moddropped, ts_moddropped, img_eff_mask, ts_eff_mask

# Perceiver IO model
# Makes use of self attention and cross attention layers
# Latent Z array
class LatentArray(layers.Layer):
    def __init__(self, num_latents, d_model, **kwargs):
        super().__init__(**kwargs)
        self.num_latents = num_latents
        self.d_model = d_model

    def build(self, input_shape):
        # Learnable [L, D] latent slots
        self.latents = self.add_weight(
            "latents",
            shape=(self.num_latents, self.d_model),
            initializer="random_normal",
        )

    def call(self, x):
        # x is any tensor we can use to infer batch size (e.g., tokens)
        batch_size = tf.shape(x)[0]
        latents = tf.broadcast_to(
            self.latents[tf.newaxis, :, :],   # [1, L, D]
            [batch_size, self.num_latents, self.d_model],  # [B, L, D]
        )
        return latents

# Latent self attention tower
class LatentTransformer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, depth, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [
            TransformerSelfAttnBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(depth)
        ]
        # self.attn = TransformerSelfAttnBlock(
        #         d_model=d_model,
        #         num_heads=num_heads,
        #         d_ff=d_ff,
        #         dropout=dropout,
        #     )

    def call(self, z, training=None):
        for blk in self.blocks:
            z = blk(z, training=training)
        return z


class TaskDecoder(layers.Layer):
    """
    Task-specific decoder:
      - Learned queries Q_t
      - Cross-attend into latent array Z
    """
    def __init__(self, num_queries, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.d_model = d_model
        self.query_emb = self.add_weight(
            "query_emb",
            shape=(num_queries, d_model),
            initializer="random_normal",
        )
        self.cross_attn = CrossAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

    def call(self, latents, training=None):
        B = tf.shape(latents)[0]
        queries = tf.broadcast_to(
            self.query_emb[None, :, :],
            (B, self.num_queries, self.d_model),
        )
        out = self.cross_attn([queries, latents], training=training)
        return out

# full perceiver style architecture
def build_multimodal_perceiver_with_multimae(
    image_shape=(72, 72, 3),
    ts_in_shape=(24, 4),
    num_classes=38,
    ts_out_len=1,
    ts_out_features=2,
    d_img=128,
    d_ts=128,
    d_model=256,
    num_latents=256,
    latent_depth=6,
    num_heads=4,
    mlp_ratio=4.0,
    dropout=0.1,
    img_patch_size=6,
    ts_patch_len=4,
    ts_patch_stride=4,
    img_mask_ratio=0.3,
    ts_mask_ratio=0.3,
    p_moddrop=0.2,
    lambda_rec=0.1,   # weight on reconstruction loss
):
    """
    Returns a Keras Model with:
      inputs:  (image, (seq_in, feat_in))
      outputs: (labels, ts_out_pred) # ts_out_pred = (ts_out_len, ts_out_features)

    And internally adds:
      λ_rec * (L_MSE_img + L_MAE_ts) on masked tokens only, MultiMAE-style.
    """

    # ------------- Inputs -------------
    image_in = keras.Input(shape=image_shape, name="image")
    ts_in = keras.Input(shape=ts_in_shape, name="ts_in")

    # ------------- Encoders -------------
    # image encoder through patch
    num_patches = (image_shape[0]//img_patch_size)**2

    # Build a small submodel using your pre-existing function   
    image_head = ImageInputHead(
        shape=image_shape,
        patch_size=img_patch_size,
        num_patches=num_patches,
        embed_dim=d_img,
    )
    
    img_tokens_raw = image_head(image_in) # [B, N_img, d_img]

    ts_encoder = PatchTSTEncoder(
        patch_len=ts_patch_len,
        stride=ts_patch_stride,
        d_ts=d_ts,
        name="ts_encoder",
    )
    ts_tokens_raw = ts_encoder(ts_in)       # [B, N_ts, d_ts]

    # project D_model
    proj_img = layers.Dense(d_model, name="proj_img")
    proj_ts = layers.Dense(d_model, name="proj_ts")

    img_content = proj_img(img_tokens_raw)  # [B, N_img, D_model]
    ts_content = proj_ts(ts_tokens_raw)     # [B, N_ts, D_model]

    # Targets for reconstruction (original embeddings)
    img_target = tf.identity(img_content)
    ts_target = tf.identity(ts_content)

    # MAE masking and drops
    mask_layer = MaskingAndModDrop(
        d_model=d_model,
        img_mask_ratio=img_mask_ratio,
        ts_mask_ratio=ts_mask_ratio,
        p_moddrop=p_moddrop,
        num_modalities=2,
        name="masking_moddrop",
    )

    img_masked, ts_masked, img_eff_mask, ts_eff_mask = mask_layer(
        [img_content, ts_content]
    )
    
    # Positional + Modality encodings 
    # Compute static #tokens for decoders & pos enc
    # Image tokens: effective downsample factor = 2*2*patch_size = 24 for (72,72)
    h_after_conv = image_shape[0] // 4
    w_after_conv = image_shape[1] // 4
    num_img_tokens = (h_after_conv // 6) * (w_after_conv // 6)

    # Time-series tokens: F * num_patches along S
    S, F = ts_in_shape
    num_ts_patches = (S - ts_patch_len) // ts_patch_stride + 1
    num_ts_tokens = F * num_ts_patches

    # Positional encodings
    img_pos_enc = PositionalEncoding(d_model=d_model, name="img_pos_enc")
    ts_pos_enc = PositionalEncoding(d_model=d_model, name="ts_pos_enc")
    
    # Modality embeddings
    mod_emb_layer = ModalityEmbedding(d_model, num_modalities=4, name="mod_emb")

    img_tokens = img_pos_enc(
        mod_emb_layer(img_masked, modality_id=tf.constant(0))
    )
    ts_tokens = ts_pos_enc(
        mod_emb_layer(ts_masked, modality_id=tf.constant(1))
    )
    # img_tokens = img_pos_enc(
    #     mod_emb_layer(img_content, modality_id=tf.constant(0))
    # )
    # ts_tokens = ts_pos_enc(
    #     mod_emb_layer(ts_content, modality_id=tf.constant(1))
    # )

    # Token concat
    tokens = layers.Concatenate(axis=1, name="concat_tokens")(
        [img_tokens, ts_tokens]
    )
    
    # Percceiver
    # latent Z array
    latent_array_layer = LatentArray(
        num_latents=num_latents, d_model=d_model, name="latent_array"
    )

    # Pass tokens as a dummy input 
    latents = latent_array_layer(tokens)  # [B, L, D_model]

    cross_encoder = CrossAttentionBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=int(d_model * mlp_ratio),
        dropout=dropout,
        name="cross_encoder",
    )
    z = cross_encoder([latents, tokens])  # [B, L, D_model]

    latent_tower = LatentTransformer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=int(d_model * mlp_ratio),
        depth=latent_depth,
        dropout=dropout,
        name="latent_tower",
    )
    # z = latent_tower(tokens)
    z = latent_tower(z)
    
    # # Debug head
    # # Flatten latents and reuse the tiny MLP style
    # z_flat = tf.keras.layers.Flatten()(z)

    # h = tf.keras.layers.Dense(128, activation="relu")(z_flat)
    # h = tf.keras.layers.Dense(128, activation="relu")(h)

    # class_logits = tf.keras.layers.Dense(num_classes, name="class_head")(h)

    # reg_out = tf.keras.layers.Dense(ts_out_len * ts_out_features)(h)
    # reg_out = tf.keras.layers.Reshape(
    #     (ts_out_len, ts_out_features), name="ts_out_pred"
    # )(reg_out)

    # ------------- Supervised task decoders -------------
    # Classification
    class_decoder = TaskDecoder(
        num_queries=1,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=int(d_model * mlp_ratio),
        dropout=dropout,
        name="class_decoder",
    )
    class_tokens = class_decoder(z)       # [B, 1, D]
    
    # now get class value
    # class_tokens = class_decoder(tokens)       # [B, 1, D]
    class_vec = class_tokens[:, 0, :]     # [B, D]
    class_logits = layers.Dense(num_classes, name="class_head")(class_vec)

    # Regression (ts_out) – one query per scalar output
    num_reg_queries = ts_out_len * ts_out_features
    reg_decoder = TaskDecoder(
        num_queries=num_reg_queries,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=int(d_model * mlp_ratio),
        dropout=dropout,
        name="reg_decoder",
    )
    reg_tokens = reg_decoder(z)  # [B, num_reg_queries, D]
    # reg_tokens = reg_decoder(tokens)  # [B, num_reg_queries, D]
    reg_out = layers.Dense(1, name="reg_head")(reg_tokens)  # [B, Q, 1]
    reg_out = tf.squeeze(reg_out, axis=-1)                  # [B, Q]
    reg_out = layers.Reshape(
        (ts_out_len, ts_out_features), name="ts_out_pred"
    )(reg_out)

    # ------------- Self-supervised reconstruction decoders (MultiMAE-style) -------------
    # Image reconstruction
    img_recon_decoder = TaskDecoder(
        num_queries=num_patches,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=int(d_model * mlp_ratio),
        dropout=dropout,
        name="img_recon_decoder",
    )
    img_recon_tokens = img_recon_decoder(z)  # [B, num_patches, D_model]

    # Time-series reconstruction
    ts_recon_decoder = TaskDecoder(
        num_queries=num_ts_tokens,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=int(d_model * mlp_ratio),
        dropout=dropout,
        name="ts_recon_decoder",
    )
    ts_recon_tokens = ts_recon_decoder(z)    # [B, N_ts, D_model]

    # ------------- Masked reconstruction losses (on embeddings) -------------
    def masked_mse(pred, target, mask_bool):
        # pred/target: [B, N, D]; mask_bool: [B, N]
        diff = pred - target
        sq = tf.reduce_sum(tf.square(diff), axis=-1)  # [B, N]
        mask_f = tf.cast(mask_bool, sq.dtype)
        num = tf.reduce_sum(sq * mask_f)
        denom = tf.reduce_sum(mask_f) + 1e-6
        return num / denom

    def masked_mae(pred, target, mask_bool):
        diff = tf.abs(pred - target)
        mae = tf.reduce_sum(diff, axis=-1)   # [B, N]
        mask_f = tf.cast(mask_bool, mae.dtype)
        num = tf.reduce_sum(mae * mask_f)
        denom = tf.reduce_sum(mask_f) + 1e-6
        return num / denom

    img_rec_loss = masked_mse(img_recon_tokens, img_target, img_eff_mask)
    ts_rec_loss = masked_mae(ts_recon_tokens, ts_target, ts_eff_mask)
    rec_loss = img_rec_loss + ts_rec_loss

    # ------------- Build model & attach reconstruction loss -------------
    model = keras.Model(
        inputs=(image_in, ts_in),
        outputs=(class_logits, reg_out),
        name="multimodal_perceiver_multimae",
    )

    # L_total = Σ_t w_t L_t + λ_rec * L_masked
    model.add_loss(lambda_rec * rec_loss)
    model.add_metric(img_rec_loss, name="img_rec_mse")
    model.add_metric(ts_rec_loss, name="ts_rec_mae")
    model.add_metric(rec_loss, name="rec_loss_total")

    return model