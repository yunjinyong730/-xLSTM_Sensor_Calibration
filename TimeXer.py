import tensorflow as tf
import numpy as np

class Patching(tf.keras.layers.Layer):
    """
    입력 시계열을 패치 단위로 분할하는 클래스입니다.
    TimeXer 논문의 Patchify (2)에 해당합니다.
    """
    def __init__(self, patch_length, name="patching", **kwargs):
        super().__init__(name=name, **kwargs)
        self.patch_length = patch_length

    def call(self, inputs):
        # inputs shape: (batch_size, sequence_length, features)
        
        # --- 수정된 부분 시작 ---
        
        # 동적인 batch_size는 tf.shape()로 가져옵니다.
        batch_size = tf.shape(inputs)[0]
        
        # sequence_length와 features는 정적 shape 정보를 사용합니다.
        # 모델 정의 시 seq_len이 고정되어 있으므로 .shape[1]로 접근 가능합니다.
        seq_len = inputs.shape[1]
        features = inputs.shape[-1]
        
        # num_patches를 파이썬 정수로 계산합니다.
        num_patches = seq_len // self.patch_length
        
        # 패치 길이에 맞게 시퀀스 길이를 조정합니다. (현재 코드에서는 불필요하지만 유지)
        seq_len_trimmed = num_patches * self.patch_length
        x = inputs[:, :seq_len_trimmed, :]
        
        # reshape 시 num_patches와 features에 정수 값을 직접 사용합니다.
        # 이렇게 하면 출력 텐서의 정적 shape이 (None, 15, 24, 1)로 올바르게 추론됩니다.
        patches = tf.reshape(
            x, [batch_size, num_patches, self.patch_length, features]
        )
        
        # --- 수정된 부분 끝 ---

        # shape: (batch_size, num_patches, patch_length, features)
        return patches
        
    def get_config(self):
        config = super().get_config()
        config.update({"patch_length": self.patch_length})
        return config

class PatchEncoder(tf.keras.layers.Layer):
    """
    패치와 위치 정보를 인코딩하여 임베딩 벡터를 생성합니다.
    TimeXer 논문의 PatchEmbed (2)와 Positional Embedding을 결합합니다.
    """
    def __init__(self, num_patches, projection_dim, name="patch_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        # patch shape: (batch_size, num_patches, patch_length, features)
        batch_size = tf.shape(patch)[0]
        
        # --- 수정된 부분 시작 ---
        
        # -1을 사용하는 대신, 평탄화될 차원의 크기를 정적 shape으로 직접 계산합니다.
        # patch.shape는 (None, 15, 24, 1)이므로 patch_shape[2]는 24, patch_shape[3]은 1입니다.
        patch_shape = patch.shape
        flattened_dim = patch_shape[2] * patch_shape[3] # 결과: 24
        
        # reshape에 계산된 값을 명시적으로 전달하여 shape 추론 문제를 해결합니다.
        patches_flattened = tf.reshape(patch, [batch_size, self.num_patches, flattened_dim])
        
        # --- 수정된 부분 끝 ---

        # 평탄화된 패치를 projection_dim 차원으로 투영합니다.
        # patches_flattened의 shape이 (None, 15, 24)로 명확하므로 Dense 레이어가 정상 동작합니다.
        projected_patches = self.projection(patches_flattened)
        
        # 위치 정보를 생성합니다.
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        
        # 투영된 패치와 위치 임베딩을 더합니다.
        encoded = projected_patches + self.position_embedding(positions)
        
        # 최종 shape: (batch_size, num_patches, projection_dim)
        return encoded
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config

class TimeXerBlock(tf.keras.layers.Layer):
    """
    TimeXer의 핵심 Transformer 블록입니다.
    - 내생 변수(Endogenous) Self-Attention
    - 외생 변수(Exogenous) -> 내생 변수 Cross-Attention
    - Feed-Forward Network
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, name="timexer_block", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        
        # 내생 변수 Self-Attention (논문 그림 2(c))
        self.endogenous_self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )
        
        # 외생 -> 내생 Cross-Attention (논문 그림 2(d))
        self.exogenous_cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )
        
        # Feed-Forward Network
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        
        # Layer Normalization
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        endogenous_tokens, exogenous_tokens = inputs
        
        # 1. 내생 변수 Self-Attention (Patch-to-Patch, Global-to-Patch 등)
        # 입력: [패치 토큰, 전역 토큰]
        # 논문 수식 (5)
        attn_output_self = self.endogenous_self_attention(
            query=endogenous_tokens,
            value=endogenous_tokens,
            key=endogenous_tokens,
        )
        # Residual Connection and Layer Normalization
        endogenous_tokens_1 = self.layernorm1(endogenous_tokens + attn_output_self)
        
        # 내생 변수 토큰에서 전역 토큰(Global Token)만 분리
        endogenous_patches = endogenous_tokens_1[:, :-1, :]
        endogenous_global = endogenous_tokens_1[:, -1:, :]

        # 2. 외생 -> 내생 Cross-Attention (Variate-to-Global)
        # Query: 내생 변수의 전역 토큰
        # Key/Value: 외생 변수의 변수 토큰
        # 논문 수식 (6)
        attn_output_cross = self.exogenous_cross_attention(
            query=endogenous_global,
            value=exogenous_tokens,
            key=exogenous_tokens,
        )
        # Residual Connection and Layer Normalization
        endogenous_global_2 = self.layernorm2(endogenous_global + attn_output_cross)
        
        # 패치 토큰과 업데이트된 전역 토큰을 다시 결합
        endogenous_tokens_2 = tf.concat([endogenous_patches, endogenous_global_2], axis=1)

        # 3. Feed-Forward Network
        ffn_output = self.ffn(endogenous_tokens_2)
        # Residual Connection and Layer Normalization
        output_tokens = self.layernorm3(endogenous_tokens_2 + ffn_output)
        
        return output_tokens
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
        })
        return config


class TimeXer(tf.keras.Model):
    """
    TimeXer 모델 전체 구조.
    - 내생/외생 변수 임베딩
    - TimeXer 블록 스택
    - 예측 헤드
    """
    def __init__(
        self,
        seq_len,
        pred_len,
        patch_length,
        embed_dim,
        num_heads,
        ff_dim,
        num_blocks,
        dropout_rate=0.1,
        name="timexer",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_length = patch_length
        self.embed_dim = embed_dim
        
        num_patches = seq_len // patch_length
        
        # --- 내생(Endogenous) 변수 처리 모듈 ---
        # 논문 그림 2(a)
        self.endogenous_patching = Patching(patch_length)
        self.endogenous_patch_encoder = PatchEncoder(num_patches, embed_dim)
        # 학습 가능한 전역 토큰(Global Token)
        self.endogenous_global_token = self.add_weight(
            shape=(1, 1, embed_dim),
            initializer="random_normal",
            trainable=True,
            name="endogenous_global_token"
        )
        
        # --- 외생(Exogenous) 변수 처리 모듈 ---
        # 논문 그림 2(b), 수식 (3)
        # 각 외생 변수 전체를 하나의 토큰으로 임베딩 (VariateEmbed)
        self.exogenous_variate_embedding = tf.keras.layers.Dense(embed_dim)
        
        # --- TimeXer 블록 ---
        self.blocks = [
            TimeXerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_blocks)
        ]
        
        # --- 예측 헤드 ---
        # 논문 수식 (8)
        self.forecasting_head = tf.keras.layers.Dense(pred_len, name="forecasting_head")
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        # 입력 텐서를 내생 변수와 외생 변수로 분리합니다.
        # (batch, seq_len, features) -> endogenous: (..., 1), exogenous: (..., features-1)
        # 첫 번째 feature를 내생 변수로, 나머지를 외생 변수로 가정합니다.
        endogenous_input = inputs[..., :1]
        exogenous_input = inputs[..., 1:]

        # 1. 내생 변수 임베딩
        # (batch, seq_len, 1) -> (batch, num_patches, patch_len, 1)
        endogenous_patches = self.endogenous_patching(endogenous_input)
        # (batch, num_patches, patch_len, 1) -> (batch, num_patches, embed_dim)
        endogenous_patch_tokens = self.endogenous_patch_encoder(endogenous_patches)
        
        # 전역 토큰을 배치 크기에 맞게 확장
        batch_size = tf.shape(endogenous_input)[0]
        global_token_expanded = tf.tile(self.endogenous_global_token, [batch_size, 1, 1])
        
        # 패치 토큰과 전역 토큰 결합
        endogenous_tokens = tf.concat([endogenous_patch_tokens, global_token_expanded], axis=1)
        
        # 2. 외생 변수 임베딩
        # 각 외생 변수를 하나의 벡터로 표현하기 위해 시간 축에 대해 평균을 냅니다.
        # 또는 다른 pooling 방법을 사용할 수 있습니다. (e.g., tf.reduce_max)
        # (batch, seq_len, num_exogenous_vars) -> (batch, num_exogenous_vars, seq_len)
        exogenous_transposed = tf.transpose(exogenous_input, perm=[0, 2, 1])
        # (batch, num_exogenous_vars, embed_dim)
        exogenous_tokens = self.exogenous_variate_embedding(exogenous_transposed)
        
        # 3. TimeXer 블록 통과
        processed_tokens = endogenous_tokens
        for block in self.blocks:
            processed_tokens = block([processed_tokens, exogenous_tokens])
            
        # 4. 예측
        # (batch, num_patches + 1, embed_dim) -> (batch, (num_patches + 1) * embed_dim)
        flattened_tokens = self.flatten(processed_tokens)
        # (batch, pred_len)
        prediction = self.forecasting_head(flattened_tokens)
        
        return prediction