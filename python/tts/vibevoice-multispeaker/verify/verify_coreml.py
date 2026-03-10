"""CoreML backend for component verification."""

from __future__ import annotations

from pathlib import Path

import coremltools as ct
import numpy as np

from verify_common import benchmark


def load_models(build_dir: Path) -> dict:
    """Load all CoreML models for verification."""
    models = {
        "diff": ct.models.MLModel(str(build_dir / "diffusion_head.mlpackage")),
        "vae": ct.models.MLModel(str(build_dir / "vae_decoder_streaming.mlpackage"),
                                compute_units=ct.ComputeUnit.CPU_AND_GPU),
        "ac_conn": ct.models.MLModel(str(build_dir / "acoustic_connector.mlpackage")),
        "sem_conn": ct.models.MLModel(str(build_dir / "semantic_connector.mlpackage")),
        "lm_head": ct.models.MLModel(str(build_dir / "lm_head.mlpackage")),
        "lm": ct.models.MLModel(str(build_dir / "lm_decoder_stateful.mlpackage")),
    }
    # Optional encoder models
    vae_enc_path = build_dir / "vae_encoder.mlpackage"
    sem_enc_path = build_dir / "semantic_encoder_streaming.mlpackage"
    models["vae_enc"] = ct.models.MLModel(str(vae_enc_path)) if vae_enc_path.exists() else None
    models["sem_enc"] = ct.models.MLModel(str(sem_enc_path),
                                          compute_units=ct.ComputeUnit.CPU_AND_GPU) if sem_enc_path.exists() else None
    return models


def test_diffusion(m: dict, noisy_np, t_np, cond_np, warmup, iters):
    out = m["diff"].predict({"noisy_latent": noisy_np, "timestep": t_np, "condition": cond_np})["predicted_noise"]
    lat = benchmark(lambda: m["diff"].predict({"noisy_latent": noisy_np, "timestep": t_np, "condition": cond_np}), warmup, iters)
    return out, lat


def test_ddpm_loop(m, noise, cond_np, timesteps, alphas, ddpm_step_fn, warmup, iters):
    from diffusion import VAE_DIM

    def _loop():
        s = noise.copy()
        for i, t in enumerate(timesteps):
            pred = m["diff"].predict({
                "noisy_latent": s,
                "timestep": np.array([float(t)], dtype=np.float32),
                "condition": cond_np,
            })["predicted_noise"]
            at = float(alphas[int(t)])
            ap = float(alphas[int(timesteps[i + 1])]) if i < len(timesteps) - 1 else 1.0
            s = ddpm_step_fn(s, pred, at, ap)
        return s

    out = _loop()
    lat = benchmark(_loop, warmup=3, iters=20)
    return out, lat


def test_vae(m, latent_np, warmup, iters):
    state = m["vae"].make_state()
    out = m["vae"].predict({"latent": latent_np}, state)["audio"]

    def _run():
        s = m["vae"].make_state()
        return m["vae"].predict({"latent": latent_np}, s)

    lat = benchmark(_run, warmup, iters)
    return out, lat


def test_vae_encoder(m, audio_np):
    if m["vae_enc"] is None:
        return None
    return m["vae_enc"].predict({"audio": audio_np})["latent"]


def _chunked_semantic_encode(model, audio_np, chunk_size=3200):
    """Run streaming semantic encoder in chunks."""
    total_samples = audio_np.shape[-1]
    state = model.make_state()
    features = []
    for start in range(0, total_samples, chunk_size):
        chunk = audio_np[:, :, start:start + chunk_size]
        if chunk.shape[-1] < chunk_size:
            chunk = np.pad(chunk, ((0, 0), (0, 0), (0, chunk_size - chunk.shape[-1])))
        out = model.predict({"audio": chunk}, state)["features"]
        features.append(out)
    return np.concatenate(features, axis=-1)


def test_semantic_encoder(m, audio_np):
    if m["sem_enc"] is None:
        return None
    return _chunked_semantic_encode(m["sem_enc"], audio_np)


def test_acoustic_connector(m, lat_np, warmup, iters):
    out = m["ac_conn"].predict({"speech_latent": lat_np})["embedding"]
    lat = benchmark(lambda: m["ac_conn"].predict({"speech_latent": lat_np}), warmup, iters)
    return out, lat


def test_semantic_connector(m, feat_np, warmup, iters):
    out = m["sem_conn"].predict({"semantic_features": feat_np})["embedding"]
    lat = benchmark(lambda: m["sem_conn"].predict({"semantic_features": feat_np}), warmup, iters)
    return out, lat


def test_lm_head(m, hidden_np, warmup, iters):
    out = m["lm_head"].predict({"hidden_state": hidden_np})["logits"]
    lat = benchmark(lambda: m["lm_head"].predict({"hidden_state": hidden_np}), warmup, iters)
    return out, lat


def test_lm_decoder(m, h_np, cos_np, sin_np, warmup, iters):
    state = m["lm"].make_state()
    out = m["lm"].predict({
        "hidden_states": h_np, "position_cos": cos_np, "position_sin": sin_np,
        "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float32),
    }, state=state)["output_hidden"]

    def _fn():
        s = m["lm"].make_state()
        m["lm"].predict({
            "hidden_states": h_np, "position_cos": cos_np, "position_sin": sin_np,
            "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float32),
        }, state=s)

    lat = benchmark(_fn, warmup, iters)
    return out, lat


def test_voice_cloning(m, ref_audio_np, ac_conn_model, sem_conn_model, sample_rate):
    """Test full voice cloning pipeline: encoders -> connectors."""
    if m["vae_enc"] is None or m["sem_enc"] is None:
        return None, None

    cml_lat = m["vae_enc"].predict({"audio": ref_audio_np})["latent"]
    cml_lat_btc = np.transpose(cml_lat, (0, 2, 1))
    cml_ac_frames = []
    for t in range(cml_lat_btc.shape[1]):
        out = m["ac_conn"].predict({"speech_latent": cml_lat_btc[:, t:t+1, :]})["embedding"]
        cml_ac_frames.append(out)
    cml_ac_emb = np.concatenate(cml_ac_frames, axis=1)

    cml_feat = _chunked_semantic_encode(m["sem_enc"], ref_audio_np)
    cml_feat_btc = np.transpose(cml_feat, (0, 2, 1))
    cml_sem_frames = []
    for t in range(cml_feat_btc.shape[1]):
        out = m["sem_conn"].predict({"semantic_features": cml_feat_btc[:, t:t+1, :]})["embedding"]
        cml_sem_frames.append(out)
    cml_sem_emb = np.concatenate(cml_sem_frames, axis=1)

    return cml_ac_emb, cml_sem_emb
