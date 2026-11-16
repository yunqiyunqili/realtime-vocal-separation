"""Evaluation metrics.

Primary: museval BSS Eval v4 SDR, win = hop = 1 s (44100 samples).
  * references/estimates evaluated jointly as [vocals, accompaniment] (SiSEC-2018 / MUSDB18
    convention, same call as museval.eval_mus_track). In v4 mode the SDR of a source depends
    only on the projection onto its own reference, so joint evaluation changes only which
    frames are NaN, not the finite values.
  * NaN: museval sets ALL metrics of a 1-s frame to NaN if any reference OR any estimate is
    all-zero in that frame. Per track we take np.nanmedian over frames (museval default);
    a track with all frames NaN gets NaN and is excluded from the cross-track median; the
    number of such tracks and of NaN frames is reported.
  * museval raises ValueError if any whole reference/estimate is all-zero (e.g. the mixture
    baseline's accompaniment = 0). We catch it, record the error string and set NaN.
  * As a fallback we also run a vocals-only museval call (nsrc=1): frames NaN only where the
    vocals reference or vocals estimate is silent.
Aux:
  * uSDR (MDX-21 "SDR"): whole track, both channels summed:
      10*log10((sum s^2 + 1e-7) / (sum (s - s_hat)^2 + 1e-7)).  A zero estimate gives 0 dB by
      construction (well-defined, not a substitute for NaN).
  * SI-SDR: whole track, stereo flattened into one vector (one shared scale factor for both
      channels), no mean removal: a = <s_hat,s>/<s,s>; 10*log10(|a s|^2 / |a s - s_hat|^2).
      NaN if the estimate or reference is all zero.
"""
import numpy as np
import museval

WIN = HOP = 44100


def usdr(ref, est, eps=1e-7):
    ref = ref.astype(np.float64); est = est.astype(np.float64)
    return float(10 * np.log10((np.sum(ref ** 2) + eps) / (np.sum((ref - est) ** 2) + eps)))


def si_sdr(ref, est):
    s = ref.astype(np.float64).ravel(); e = est.astype(np.float64).ravel()
    ss = np.dot(s, s)
    if ss == 0 or not np.any(e):
        return float("nan")
    a = np.dot(e, s) / ss
    t = a * s
    num, den = np.dot(t, t), np.dot(t - e, t - e)
    if num == 0 or den == 0:
        return float("nan")
    return float(10 * np.log10(num / den))


def _museval(refs, ests):
    try:
        sdr, isr, sir, sar = museval.evaluate(np.stack(refs), np.stack(ests), win=WIN, hop=HOP, mode="v4")
        return sdr, None
    except Exception as e:  # recorded, never silently replaced by 0
        return None, f"{type(e).__name__}: {e}"


def _frame_summary(frames):
    frames = np.asarray(frames, dtype=np.float64)
    n_nan = int(np.isnan(frames).sum())
    med = float(np.nanmedian(frames)) if n_nan < frames.size else float("nan")
    return med, n_nan, int(frames.size)


def evaluate_track(mix, vocals_ref, acc_ref, vocals_est):
    """All arrays (T, 2) float. accompaniment estimate = mixture - vocals_est."""
    T = min(len(mix), len(vocals_ref), len(vocals_est))
    mix, vocals_ref, acc_ref, vocals_est = mix[:T], vocals_ref[:T], acc_ref[:T], vocals_est[:T]
    acc_est = mix - vocals_est
    r = {}
    sdr, err = _museval([vocals_ref, acc_ref], [vocals_est, acc_est])
    if sdr is None:
        r.update(museval_joint_error=err, sdr_vocals=float("nan"), sdr_acc=float("nan"),
                 nan_frames_joint=None, n_frames=None)
    else:
        v, nv, n = _frame_summary(sdr[0]); a, na, _ = _frame_summary(sdr[1])
        r.update(museval_joint_error=None, sdr_vocals=v, sdr_acc=a, nan_frames_joint=nv, n_frames=n)
    sdr1, err1 = _museval([vocals_ref], [vocals_est])
    if sdr1 is None:
        r.update(museval_vonly_error=err1, sdr_vocals_vonly=float("nan"), nan_frames_vonly=None)
    else:
        v1, nv1, n1 = _frame_summary(sdr1[0])
        r.update(museval_vonly_error=None, sdr_vocals_vonly=v1, nan_frames_vonly=nv1, n_frames=r.get("n_frames") or n1)
    r.update(usdr_vocals=usdr(vocals_ref, vocals_est), usdr_acc=usdr(acc_ref, acc_est),
             sisdr_vocals=si_sdr(vocals_ref, vocals_est), sisdr_acc=si_sdr(acc_ref, acc_est),
             n_samples=int(T))
    return r


def summarize(rows):
    """rows: list of per-track dicts. Medians for museval SDR, mean+median for uSDR/SI-SDR."""
    import pandas as pd
    df = pd.DataFrame(rows)
    s = {"n_tracks": len(df)}
    for k in ["sdr_vocals", "sdr_acc", "sdr_vocals_vonly"]:
        col = df[k].astype(float)
        s[k + "_median_of_track_medians"] = float(col.median()) if col.notna().any() else float("nan")
        s[k + "_n_nan_tracks"] = int(col.isna().sum())
    for k in ["usdr_vocals", "usdr_acc", "sisdr_vocals", "sisdr_acc"]:
        col = df[k].astype(float)
        s[k + "_mean"] = float(col.mean()) if col.notna().any() else float("nan")
        s[k + "_median"] = float(col.median()) if col.notna().any() else float("nan")
        s[k + "_n_nan_tracks"] = int(col.isna().sum())
    errs = df["museval_joint_error"].dropna().unique().tolist()
    s["museval_joint_errors"] = errs
    s["museval_vonly_errors"] = df["museval_vonly_error"].dropna().unique().tolist()
    return s
