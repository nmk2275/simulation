# bb84_research_dashboard.py
"""
BB84 QKD Research Dashboard (Streamlit)

Features:
- Realistic noise model (Model B):
    * photon loss removes qubits
    * dark counts produce random detections in lost windows (or optional simple flip model)
    * channel & detector noise flip surviving detections
    * optional Eve intercept-resend (on/off)
- Monte-Carlo averaging (mean + 95% CI)
- Error attribution (channel / detector / dark / Eve)
- Bar chart: counts surviving after each stage
- Professional QBER vs Channel Noise plot
- Export results CSV
"""

import streamlit as st
import random
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import io
import math
from statistics import mean, stdev

# -------------------------
# Data classes & utilities
# -------------------------
@dataclass
class Qubit:
    bit: int
    basis: str  # 'R' or 'D'

@dataclass
class Measurement:
    bit: int
    basis: str

# -------------------------
# Simulator (Model B)
# -------------------------
class BB84Simulator:
    def __init__(self,
                 initial_qubits: int = 5000,
                 channel_noise: float = 0.05,
                 detector_noise: float = 0.02,
                 dark_count_prob: float = 0.00004,
                 photon_loss: float = 0.10,
                 attack_influence: float = 0.0,
                 ee_sample_fraction: float = 0.10,
                 error_threshold: float = 0.11,
                 block_size: int = 16,
                 max_recon_rounds: int = 6,
                 pa_shrink_fraction: float = 0.07,
                 dark_count_mode: str = "on_loss",  # "on_loss" or "as_flip"
                 rng_seed: int = None):
        self.initial_qubits = initial_qubits
        self.channel_noise = channel_noise
        self.detector_noise = detector_noise
        self.dark_count_prob = dark_count_prob
        self.photon_loss = photon_loss
        self.attack_influence = attack_influence
        self.ee_sample_fraction = ee_sample_fraction
        self.error_threshold = error_threshold
        self.block_size = block_size
        self.max_recon_rounds = max_recon_rounds
        self.pa_shrink_fraction = pa_shrink_fraction
        self.dark_count_mode = dark_count_mode
        if rng_seed is not None:
            random.seed(rng_seed)

    def generate_qubits(self) -> List[Qubit]:
        return [Qubit(random.randint(0,1), random.choice(['R','D'])) for _ in range(self.initial_qubits)]

    def eve_intercept_resend(self, qubit: Qubit) -> Qubit:
        eve_basis = random.choice(['R','D'])
        measured = qubit.bit if eve_basis == qubit.basis else random.randint(0,1)
        return Qubit(measured, eve_basis)

    def transmit_and_measure(self, qubits: List[Qubit]) -> Tuple[List[Tuple[int, Measurement, str]], Dict[str,int]]:
        """
        Returns:
            detections: list of tuples (original_index, Measurement, detection_origin)
                        detection_origin in {'survived', 'dark_count'}
            stats: dictionary of counts and error-attribution counters placeholders (filled later)
        """
        detections = []
        eve_count = 0
        stats = {
            'generated': len(qubits),
            'survived_photon': 0,
            'dark_count_detections': 0,
            'total_detections': 0,
            'eve_count': 0
        }

        for idx, q in enumerate(qubits):
            q_after = q
            # Eve intercept/resend (if active)
            if random.random() < self.attack_influence:
                q_after = self.eve_intercept_resend(q)
                eve_count += 1

            # Photon loss
            if random.random() < self.photon_loss:
                # photon is lost: maybe a dark count creates a detection
                if self.dark_count_mode == "on_loss":
                    if random.random() < self.dark_count_prob:
                        rand_bit = random.randint(0,1)
                        rand_basis = random.choice(['R','D'])
                        detections.append((idx, Measurement(rand_bit, rand_basis), 'dark_count'))
                        stats['dark_count_detections'] += 1
                        stats['total_detections'] += 1
                else:
                    # dark_count_mode == "as_flip" => treat dark as tiny flip (ignore here because lost)
                    pass
                continue  # next time slot

            # Photon survived
            stats['survived_photon'] += 1
            recv_basis = random.choice(['R','D'])
            if recv_basis == q_after.basis:
                measured_bit = q_after.bit
            else:
                measured_bit = random.randint(0,1)

            # Channel noise flips are applied to surviving photons
            if random.random() < self.channel_noise:
                measured_bit ^= 1

            # Detector noise flips measured outcome
            if random.random() < self.detector_noise:
                measured_bit ^= 1

            # If dark_count_mode == "as_flip", optionally treat dark_count_prob as extra small flip on survivors
            if self.dark_count_mode == "as_flip":
                if random.random() < self.dark_count_prob:
                    measured_bit ^= 1

            detections.append((idx, Measurement(measured_bit, recv_basis), 'survived'))
            stats['total_detections'] += 1

        stats['eve_count'] = eve_count
        return detections, stats

    def sift_key(self, qubits: List[Qubit], detections: List[Tuple[int, Measurement, str]]) -> Tuple[List[int], List[int], List[str]]:
        """Return sender_bits, receiver_bits, detection_origins_for_these_sifted"""
        s_bits = []
        r_bits = []
        origins = []
        for idx, meas, origin in detections:
            alice_qubit = qubits[idx]
            if alice_qubit.basis == meas.basis:
                s_bits.append(alice_qubit.bit)
                r_bits.append(meas.bit)
                origins.append(origin)
        return s_bits, r_bits, origins

    def error_estimation(self, s_bits: List[int], r_bits: List[int]) -> Tuple[float, List[int], List[int], int]:
        n = len(s_bits)
        if n == 0:
            return 0.0, [], [], 0
        sample_size = max(1, int(self.ee_sample_fraction * n))
        indices = random.sample(range(n), sample_size)
        sample_s = [s_bits[i] for i in indices]
        sample_r = [r_bits[i] for i in indices]
        errors = sum(1 for a, b in zip(sample_s, sample_r) if a != b)
        qber = errors / sample_size
        remaining_s = [b for i, b in enumerate(s_bits) if i not in indices]
        remaining_r = [b for i, b in enumerate(r_bits) if i not in indices]
        return qber, remaining_s, remaining_r, sample_size

    def parity_block_reconcile(self, s_bits: List[int], r_bits: List[int]) -> Tuple[List[int], List[int]]:
        s = s_bits.copy()
        r = r_bits.copy()
        if len(s) == 0:
            return s, r
        rounds = 0
        while rounds < self.max_recon_rounds:
            changed = False
            num_blocks = (len(s) + self.block_size - 1) // self.block_size
            for b in range(num_blocks):
                start = b * self.block_size
                end = min(start + self.block_size, len(s))
                s_block = s[start:end]
                r_block = r[start:end]
                if not s_block:
                    continue
                if sum(s_block) % 2 != sum(r_block) % 2:
                    lo, hi = start, end - 1
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if sum(s[lo:mid+1]) % 2 != sum(r[lo:mid+1]) % 2:
                            hi = mid
                        else:
                            lo = mid + 1
                    r[lo] ^= 1
                    changed = True
            rounds += 1
            if not changed:
                break
        return s, r

    def privacy_amplification(self, reconciled_bits: List[int]) -> Tuple[str, int]:
        if not reconciled_bits:
            return "", 0
        bitstring = ''.join(str(b) for b in reconciled_bits)
        h = hashlib.sha256(bitstring.encode('utf-8')).hexdigest()
        final_hex_len = max(2, int(len(h) * self.pa_shrink_fraction))
        final_hex = h[:final_hex_len]
        final_bits = final_hex_len * 4
        return final_hex, final_bits

    def attribute_errors(self, s_sample: List[int], r_sample: List[int], origins_sample: List[str],
                         qubits_indices_sample: List[int], qubits: List[Qubit]) -> Dict[str,int]:
        """
        Given the sampled indices (for EE) and their origins, estimate how many sampled errors were caused by which source.
        This attribution is approximate and heuristic: when origin == 'survived' we attribute to channel/detector flips probabilistically
        by checking exact bitwise flips compared to original qubit.
        For origin == 'dark_count' we attribute to 'dark'.
        """
        # This function is used only to get a breakdown for sampled portion; main attribution is done below per run.
        counts = {'channel':0, 'detector':0, 'dark':0, 'eve':0, 'unknown':0}
        return counts  # placeholder; main attribution uses per-event comparison in run()

    def run_single(self) -> Dict:
        # Single-run simulation returning many detailed fields
        qubits = self.generate_qubits()
        detections, stats = self.transmit_and_measure(qubits)
        s_bits, r_bits, origins = self.sift_key(qubits, detections)
        qber, s_remaining, r_remaining, sample_size = self.error_estimation(s_bits, r_bits)
        reconciled_s, reconciled_r = self.parity_block_reconcile(s_remaining, r_remaining)
        final_hex, final_bits = self.privacy_amplification(reconciled_s)
        disagreements = sum(1 for a, b in zip(reconciled_s, reconciled_r) if a != b)
        aborted = qber > self.error_threshold

        # Error attribution across all sifted bits (not just sampled)
        attrib = {'channel':0, 'detector':0, 'dark':0, 'eve':0, 'other':0}
        # We'll re-simulate origins mapping for attributions: use detections aligned with sifted indices
        # Build mapping from sifted index to (orig index, origin label)
        sifted_info = []
        for idx, (qidx, meas, origin) in enumerate(detections):
            alice = qubits[qidx]
            if alice.basis == meas.basis:
                sifted_info.append((qidx, origin, alice.bit, meas.bit))

        # Attribution heuristic: when measured bit != alice bit:
        for (qidx, origin, alice_bit, bob_bit) in sifted_info:
            if alice_bit == bob_bit:
                continue
            # if origin is dark_count, attribute to dark
            if origin == 'dark_count':
                attrib['dark'] += 1
                continue
            # origin == 'survived' -> the error could be channel, detector, or Eve (if Eve was active)
            # We cannot perfectly separate channel vs detector after the fact; but we can attribute fractionally:
            # heuristic: attribute proportionally to their probabilities
            p_c = self.channel_noise
            p_t = self.detector_noise
            # if Eve active, add small attribution to Eve based on attack_influence
            if self.attack_influence > 0:
                # If Eve intercepted that qubit (we don't store per-qubit eve flag), assume fraction attack_influence of errors due to Eve.
                # This is approximate.
                p_e = self.attack_influence
            else:
                p_e = 0.0
            # normalize
            weights = {'channel': p_c, 'detector': p_t, 'eve': p_e}
            s = p_c + p_t + p_e
            if s == 0:
                attrib['other'] += 1
            else:
                # add fractional counts
                for k in ['channel','detector','eve']:
                    attrib[k] += (weights[k] / s)
        # Round fractional counts to nearest ints for display
        for k in attrib:
            attrib[k] = round(attrib[k], 2)

        counts = {
            'generated': stats['generated'],
            'survived_photon': stats['survived_photon'],
            'total_detections': stats['total_detections'],
            'sifted': len(s_bits),
            'after_EE': len(s_remaining),
            'after_KR': len(reconciled_s),
            'final_bits': final_bits
        }

        return {
            'qber': qber,
            'sample_size': sample_size,
            'raw_preview': ''.join(str(b) for b in s_bits[:50]),
            'counts': counts,
            'final_hex': final_hex,
            'final_bits': final_bits,
            'disagreements': disagreements,
            'aborted': aborted,
            'eve_count': stats['eve_count'],
            'attrib': attrib
        }

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Simulation", layout="wide")
st.title("🔬 Alice-Bob Simulation")

with st.expander("Quick notes (click to expand)", expanded=False):
    st.markdown("""
   1️⃣ Protocol Overview**
- Alice prepares qubits randomly in either the **Rectilinear (R)** or **Diagonal (D)** basis with random bits.
- Qubits are transmitted over a channel to Bob, who measures in a randomly chosen basis.
- Only the bits where Alice and Bob used the same basis are kept (**sifted key**).
- The **Quantum Bit Error Rate (QBER)** quantifies mismatches in the sifted key and signals possible eavesdropping or noise.

2️⃣ Simulation Model (Model B – realistic)**
- **Photon loss (`photon_loss`)**: fraction of photons lost in the channel; lost photons may trigger dark counts.
- **Dark counts (`dark_count_prob`)**: random detection events when no photon arrives; modeled as:
    - `"on_loss"`: only on lost photons.
    - `"as_flip"`: small probability of flipping surviving photons.
- **Channel noise (`channel_noise`)**: probability of flipping a surviving photon due to the channel.
- **Detector noise (`detector_noise`)**: probability of misreading the measured bit.
- **Eve intercept-resend (`attack_influence`)**: fraction of qubits intercepted; measured in random basis and resent.
- **Error Estimation (EE)**: a fraction of the sifted key (`ee_sample_fraction`) is revealed to estimate **QBER**.

3️⃣ Key Calculations**
- **Sifted key selection**: only qubits with matching bases are kept.
- **QBER (sampled)**:
\[
\text{QBER} = \frac{\text{\# mismatched bits in sample}}{\text{sample size}}
\]
- **Parity block reconciliation**: iterative parity checks per block (`block_size`) to correct errors without revealing key bits.
- **Privacy amplification**: hash the reconciled key, shrink by `pa_shrink_fraction` to remove Eve’s information:
\[
\text{final key length} \approx \text{len(S)} \times \text{pa_shrink_fraction}
\]
- **Monte-Carlo averaging**: run N independent simulations to compute mean QBER, 95% confidence interval, and key statistics.

4️⃣ Error Attribution**
- Approximate source attribution for QBER:
    - **Channel flips**: due to `channel_noise`
    - **Detector flips**: due to `detector_noise`
    - **Dark counts**: from lost photons
    - **Eve**: fraction of errors attributed to intercept-resend attacks
- Attribution is heuristic and proportional to parameter probabilities.

5️⃣ Parameters Summary**
| Parameter | Description | Default / Typical Range |
|-----------|-------------|------------------------|
| `initial_qubits` | Number of qubits sent by Alice | 5000 |
| `channel_noise` | Channel-induced bit flip probability | 0–5% |
| `detector_noise` | Probability detector flips measured bit | 0–2% |
| `dark_count_prob` | Probability of random detection per window | ~0.0004 |
| `photon_loss` | Fraction of photons lost | 0–90% |
| `ee_sample_fraction` | Fraction of sifted key revealed for QBER | 10% |
| `block_size` | Bits per block in parity reconciliation | 16 |
| `max_recon_rounds` | Max iterative reconciliation rounds | 6 |
| `pa_shrink_fraction` | Fraction of SHA-256 hash used for final key | 0.07 |
| `attack_influence` | Fraction of qubits intercepted by Eve | 0–1 |

6️⃣ Notes for Researchers**
- Increasing `initial_qubits` reduces statistical fluctuations.
- Smaller EE fraction reduces key consumption but increases uncertainty in QBER.
- High photon loss or noise increases QBER; aborted if QBER > 11%.
- Monte-Carlo averaging helps observe trends and smooth stochastic variations.
- Use the plots (QBER vs channel noise, stage-wise counts) to study trade-offs and protocol robustness.

    """)

# Sidebar controls
st.sidebar.header("Simulation controls")

initial_qubits = st.sidebar.number_input("Initial qubits", min_value=200, max_value=200000, value=5000, step=100)
channel_noise_pct = st.sidebar.slider("Channel noise (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.1,
                                      help="Typical experimental channel noise; default range 0-5%")
detector_noise_pct = st.sidebar.slider("Detector noise (%)", min_value=0.0, max_value=2.0, value=0.5, step=0.05,
                                       help="Detector misread probability (0-2%)")
dark_count_prob = st.sidebar.slider("Dark counts (probability per window)", min_value=0.0, max_value=0.01, value=0.0004, step=0.0001,
                                    help="Small probability per time window (e.g., 0.0004 ~ 0.04%)")
photon_loss_pct = st.sidebar.slider("Photon loss (%)", min_value=0.0, max_value=90.0, value=5.0, step=0.5,
                                    help="Fraction of photons lost in channel (0-90%)")
ee_sample_fraction = st.sidebar.slider("EE sample fraction (%)", min_value=1, max_value=30, value=10, step=1,
                                       help="Fraction of sifted key revealed for QBER estimation (1-30%)")
dark_count_mode = st.sidebar.selectbox("Dark count model", options=["on_loss", "as_flip"],
                                       help="'on_loss' = dark counts generate random detection when photon lost (realistic). 'as_flip' = small flip on survivors (simple)")
eve_on = st.sidebar.checkbox("Eve intercept-resend (on = intercept all)", value=False)
attack_influence = 1.0 if eve_on else 0.0
mc_runs = st.sidebar.number_input("Monte-Carlo runs (N)", min_value=1, max_value=200, value=30, step=1)
seed_input = st.sidebar.number_input("RNG seed (0 for random)", min_value=0, max_value=2**31-1, value=0, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("Advanced options:")
block_size = st.sidebar.number_input("Reconciliation block size", min_value=8, max_value=4096, value=16, step=8)
max_recon_rounds = st.sidebar.number_input("Max reconciliation rounds", min_value=1, max_value=50, value=6, step=1)
pa_shrink_fraction = st.sidebar.slider("PA shrink fraction (fraction of hash used)", min_value=0.01, max_value=0.3, value=0.07, step=0.01)

# run button
run_button = st.sidebar.button("Run Simulation")

# Info column
st.markdown("### Controls summary")
st.write({
    "Initial qubits": initial_qubits,
    "Channel noise (%)": channel_noise_pct,
    "Detector noise (%)": detector_noise_pct,
    "Dark count prob (per window)": dark_count_prob,
    "Photon loss (%)": photon_loss_pct,
    "EE sample fraction (%)": ee_sample_fraction,
    "Dark-count model": dark_count_mode,
    "Eve on": eve_on,
    "Monte-Carlo runs": mc_runs
})

# Placeholder for outputs
output_placeholder = st.empty()

# Helper: run MC and collect results
def run_monte_carlo(params, mc_runs, seed_input):
    results = []
    # set a reproducible seed if requested, otherwise vary runs
    base_seed = None
    if seed_input != 0:
        base_seed = int(seed_input)
    for i in range(mc_runs):
        seed = base_seed + i if base_seed is not None else None
        sim = BB84Simulator(
            initial_qubits=params['initial_qubits'],
            channel_noise=params['channel_noise'],
            detector_noise=params['detector_noise'],
            dark_count_prob=params['dark_count_prob'],
            photon_loss=params['photon_loss'],
            attack_influence=params['attack_influence'],
            ee_sample_fraction=params['ee_sample_fraction'],
            block_size=params['block_size'],
            max_recon_rounds=params['max_recon_rounds'],
            pa_shrink_fraction=params['pa_shrink_fraction'],
            dark_count_mode=params['dark_count_mode'],
            rng_seed=seed
        )
        single = sim.run_single()
        # include run-level params and attribution
        row = {
            'run': i+1,
            'qber': single['qber'],
            'sample_size': single['sample_size'],
            'raw_preview': single['raw_preview'],
            'sifted_len': single['counts']['sifted'],
            'after_EE': single['counts']['after_EE'],
            'after_KR': single['counts']['after_KR'],
            'final_bits': single['final_bits'],
            'disagreements': single['disagreements'],
            'aborted': single['aborted'],
            'eve_count': single['eve_count'],
            'attrib_channel': single['attrib']['channel'],
            'attrib_detector': single['attrib']['detector'],
            'attrib_dark': single['attrib']['dark'],
            'attrib_eve': single['attrib']['eve']
        }
        results.append(row)
    return pd.DataFrame(results)

# Main action
if run_button:
    params = {
        'initial_qubits': initial_qubits,
        'channel_noise': channel_noise_pct / 100.0,
        'detector_noise': detector_noise_pct / 100.0,
        'dark_count_prob': dark_count_prob,
        'photon_loss': photon_loss_pct / 100.0,
        'attack_influence': attack_influence,
        'ee_sample_fraction': ee_sample_fraction / 100.0,
        'block_size': block_size,
        'max_recon_rounds': max_recon_rounds,
        'pa_shrink_fraction': pa_shrink_fraction,
        'dark_count_mode': dark_count_mode
    }

    # Run Monte Carlo (with progress bar)
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Running simulations...")
    df_results = run_monte_carlo(params, mc_runs, seed_input)
    progress_bar.progress(100)
    status_text.text("Simulation complete.")

    # Summary stats
    mean_qber = df_results['qber'].mean()
    se_qber = df_results['qber'].std(ddof=1) / math.sqrt(len(df_results)) if len(df_results) > 1 else 0.0
    ci95_low = mean_qber - 1.96 * se_qber
    ci95_high = mean_qber + 1.96 * se_qber

    # Attribution averages
    attrib_channel = df_results['attrib_channel'].mean()
    attrib_detector = df_results['attrib_detector'].mean()
    attrib_dark = df_results['attrib_dark'].mean()
    attrib_eve = df_results['attrib_eve'].mean()

    # Display stage-wise info with nicer layout
    with output_placeholder.container():
        st.header("Simulation results")
        col1, col2, col3 = st.columns([1.2,1,1])
        col1.metric("QBER (mean)", f"{mean_qber:.4f}", delta=f"95% CI [{ci95_low:.4f}, {ci95_high:.4f}]")
        col1.write(f"Runs: {mc_runs}")
        col2.metric("Sifted length (mean)", f"{int(df_results['sifted_len'].mean()):,}")
        col2.metric("Final key bits (mean)", f"{int(df_results['final_bits'].mean()):,}")
        col3.metric("Aborted runs", f"{df_results['aborted'].sum()} / {mc_runs}")
        col3.metric("Eve interceptions (mean)", f"{df_results['eve_count'].mean():.2f}")

        st.markdown("### Error attribution (average over runs)")
        attrib_df = pd.DataFrame({
            'source': ['channel', 'detector', 'dark', 'eve'],
            'avg_count': [attrib_channel, attrib_detector, attrib_dark, attrib_eve]
        })
        st.bar_chart(data=attrib_df.set_index('source'))

        st.markdown("### Raw example (first run preview)")
        st.text_area("Sifted key (first 50 bits) [run 1]", df_results.loc[0, 'raw_preview'], height=80)

        st.markdown("### Full results table (each Monte-Carlo run)")
        st.dataframe(df_results.style.format({
            'qber': "{:.4f}",
            'attrib_channel': "{:.2f}",
            'attrib_detector': "{:.2f}",
            'attrib_dark': "{:.2f}",
            'attrib_eve': "{:.2f}"
        }), height=240)

        # Bar chart counts averaged across runs: we ran each run independently; we can re-run single runs to get counts properly or approximate using the sample
        st.markdown("### QBER vs Channel Noise (sweep) — professional plot")
        # We'll compute a sweep: vary channel noise from 0 to chosen channel noise (fine grid)
        max_noise_pct = channel_noise_pct
        grid_points = 25
        noise_grid = np.linspace(0.0, max_noise_pct/100.0, grid_points)
        sweep_qbers = []
        # For speed: for each grid point run a small MC (min(5, mc_runs)) and average
        inner_runs = min(5, max(1, mc_runs // 6))
        sweep_progress = st.progress(0)
        for i, nv in enumerate(noise_grid):
            params_local = params.copy()
            params_local['channel_noise'] = float(nv)
            df_local = run_monte_carlo(params_local, inner_runs, seed_input)
            sweep_qbers.append(df_local['qber'].mean())
            sweep_progress.progress(int((i+1)/len(noise_grid)*100))
        sweep_progress.empty()

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(noise_grid * 100.0, sweep_qbers, linewidth=2.4, marker='o', markersize=5)
        ax.axhline(0.11, color='red', linestyle='--', linewidth=1.2, label='Abort threshold (11%)')
        ax.set_xlabel("Channel noise (%)")
        ax.set_ylabel("QBER")
        ax.set_title("QBER vs Channel Noise (averaged small MC per point)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        st.markdown("### Bar chart: average surviving counts per run")
        # To get approximate counts, rerun a small set of runs and average counts (this is costly if mc_runs large; we do a sampled approach)
        counts_runs = []
        sample_count_runs = min(10, mc_runs)
        for i in range(sample_count_runs):
            seed = (seed_input + 1000 + i) if seed_input != 0 else None
            sim_tmp = BB84Simulator(
                initial_qubits=initial_qubits,
                channel_noise=params['channel_noise'],
                detector_noise=params['detector_noise'],
                dark_count_prob=params['dark_count_prob'],
                photon_loss=params['photon_loss'],
                attack_influence=params['attack_influence'],
                ee_sample_fraction=params['ee_sample_fraction'],
                block_size=params['block_size'],
                max_recon_rounds=params['max_recon_rounds'],
                pa_shrink_fraction=params['pa_shrink_fraction'],
                dark_count_mode=params['dark_count_mode'],
                rng_seed=seed
            )
            # Run single but extract counts
            single = sim_tmp.run_single()
            # reconstruct counts by calling a fresh sim - but we used run_single which includes counts in 'counts' returned earlier,
            # but for speed we didn't capture those; instead we use the previously collected df_results for rough numbers.
            counts_runs.append({
                'sifted': single['counts']['sifted'] if 'counts' in single else None,
                'after_EE': single['counts']['after_EE'] if 'counts' in single else None,
                'after_KR': single['counts']['after_KR'] if 'counts' in single else None
            })
        # if counts_runs empty or None, fallback to df_results means
        # Build bar values from df_results aggregated approximations
        avg_sifted = int(df_results['sifted_len'].mean())
        avg_after_EE = int(df_results['after_EE'].mean()) if 'after_EE' in df_results.columns else avg_sifted
        avg_after_KR = int(df_results['after_KR'].mean()) if 'after_KR' in df_results.columns else avg_after_EE
        bar_labels = ['Sifted (mean)', 'After EE (mean)', 'After KR (mean)', 'Final bits (mean)']
        bar_values = [avg_sifted, avg_after_EE, avg_after_KR, int(df_results['final_bits'].mean())]
        fig2, ax2 = plt.subplots(figsize=(10,4))
        colors = ['#2b7fbf', '#f6c85f', '#9b5de5', '#2ec4b6']
        ax2.bar(bar_labels, bar_values, color=colors)
        ax2.set_title("Average counts after each stage (approx.)")
        ax2.set_ylabel("Count")
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
        st.pyplot(fig2)

        # CSV export
        csv_buffer = io.StringIO()
        df_results.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()
        st.download_button("Download results CSV", data=csv_bytes, file_name="bb84_mc_results.csv", mime="text/csv")

        st.success("Simulation complete. Use the Download button to get raw results.")
