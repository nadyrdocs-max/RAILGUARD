import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp
from scipy.stats import kurtosis, skew
from scipy.signal import welch, stft, find_peaks
import pywt

base_dir = r"C:\Users\saule\OneDrive\Рабочий стол\Projects\RAILGUARD"

def simulate_wheel_flat(flat_length_mm=0, speed_mps=10, T=2.0, fs=10000, noise_lvl=0.1):
    # Wheel and bogie mass (lab stand values)
    mw = 0.7
    mb = 2.0
    
    # Suspension stiffness and damping
    ks = 2e6
    cs = 5000
    # Hertz contact coefficient (simplified)
    kh = 1.5e8
    # Wheel radius (lab stand: 60mm)
    R = 0.060
    # Impact coefficient
    impact_coeff = 0.3

    # Время симуляции
    t_span = (0, T)
    t_eval = np.linspace(0, T, int(T * fs))

    # Геометрия плоской площадки
    L = flat_length_mm / 1000.0  # Длина flat в метрах
    # Угол плоской площадки (в радианах)
    alpha = 2 * np.arcsin(L / (2 * R)) if L > 0 and L < 2*R else 0
    # Угловая скорость колеса (рад/с) - в реальном стенде измеряется энкодером/датчиком Холла
    omega = speed_mps / R

    def get_contact_force_and_state(t):
        """
        Возвращает силу контакта и состояние контакта:
        0 - нормальный контакт
        1 - ведущий край удара
        2 - контакт с плоской площадкой (уменьшенная жесткость)
        3 - trailing edge impact
        """
        theta = (omega * t) % (2 * np.pi)  # Текущий угол положения колеса

        # Нормализуем позицию относительно начала flat зоны
        # Flat зона находится от -alpha/2 до alpha/2 относительно вершины колеса
        # Но проще: считаем, что flat зона начинается при theta = 0
        angle_norm = theta % (2 * np.pi)

        if L == 0:  # Нет дефекта - нормальный контакт
            delta = 0.0001  # Маленький начальный зазор для предотвращения деления на ноль
            contact_state = 0
            velocity_factor = 1.0
        else:
            # Проверяем, находится ли колесо в зоне flat
            if angle_norm < alpha:  # Внутри flat зоны
                # Внутри flat зоны - уменьшенная жесткость (плоская поверхность не обеспечивает упругое восстановление)
                # Используемmuch smaller stiffness to simulate lack of elastic recovery
                delta = L**2 / (8 * R)  # Геометрический прогиб из-за flat
                contact_state = 2  # Flat tread contact
                velocity_factor = 0.1  # Очень маленький коэффициент скорости для flat области
            else:
                # Вне flat зоны - проверяем proximity к краям для эффекта удара
                # Удар происходит при входе и выходе из flat зоны
                dist_to_leading_edge = min((angle_norm - alpha) % (2 * np.pi),
                                         (alpha - angle_norm) % (2 * np.pi))
                dist_to_trailing_edge = min(angle_norm % (2 * np.pi),
                                          (2 * np.pi - angle_norm) % (2 * np.pi))

                # Если близко к краю flat зоны (< 10% от alpha), моделируем удар
                impact_zone = 0.1 * alpha
                if dist_to_leading_edge < impact_zone or dist_to_trailing_edge < impact_zone:
                    # Находимся в зоне удара
                    delta = 0.0001  # Базовый зазор
                    # Вычисляем относительную скорость удара
                    # Упрощенно: предполагаем максимальную скорость при ударе
                    v_impact = speed_mps * 0.5  # Приблизительная скорость компонента нормального к поверхности
                    velocity_factor = 1.0 + impact_coeff * (v_impact / speed_mps)  # Увеличиваем силу удара
                    # Определяем тип удара
                    if dist_to_leading_edge < impact_zone:
                        contact_state = 1  # Leading edge impact
                    else:
                        contact_state = 3  # Trailing edge impact
                else:
                    # Нормальный контакт вне зоны удара
                    delta = 0.0001
                    contact_state = 0
                    velocity_factor = 1.0

        # Закон Герца с учетом скоростного эффекта для ударов
        base_force = kh * (max(delta, 0.0001)**1.5)
        contact_force = base_force * velocity_factor

        return contact_force, contact_state

    # 2-DOF system: wheel and bogie
    def dynamics(t, y):
        xw, vw, xb, vb = y
        f_contact, _ = get_contact_force_and_state(t)

        # Suspension force
        f_susp = ks * (xw - xb) + cs * (vw - vb)

        # Accelerations
        aw = (f_contact - f_susp) / mw  # wheel acceleration
        ab = f_susp / mb                # bogie acceleration

        return [vw, aw, vb, ab]

    # Solve ODE system
    sol = solve_ivp(dynamics, t_span, [0, 0, 0, 0], t_eval=t_eval, method='RK45')

    # Extract wheel acceleration
    accel_wheel = []
    contact_states = []
    for i in range(len(sol.t)):
        xw, vw, xb, vb = sol.y[:, i]
        f_contact, contact_state = get_contact_force_and_state(sol.t[i])
        f_susp = ks * (xw - xb) + cs * (vw - vb)
        aw = (f_contact - f_susp) / mw
        accel_wheel.append(aw)
        contact_states.append(contact_state)

    accel_wheel = np.array(accel_wheel)
    contact_states = np.array(contact_states)

    # Add sensor noise
    noise = np.random.normal(0, noise_lvl, len(accel_wheel))
    accel_measured = accel_wheel + noise

    return sol.t, accel_measured, accel_wheel, contact_states

def extract_features(signal, fs=10000):
    # Basic features
    abs_sig = np.abs(signal)
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(abs_sig)
    mean_abs = np.mean(abs_sig)

    # Statistical features
    crest_factor = peak / rms if rms != 0 else 0
    # Kurtosis (using scipy's excess kurtosis, so add 3 for true value)
    kurt = kurtosis(signal) + 3
    skewness = skew(signal)

    # Shape and impulse factors
    shape_factor = rms / mean_abs if mean_abs != 0 else 0
    impulse_factor = peak / mean_abs if mean_abs != 0 else 0

    # Spectral features
    # Spectral entropy
    freqs, psd = welch(signal, fs=fs, nperseg=min(1024, len(signal)//4))
    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    # Band energy (0-200Hz, 200-1000Hz, 1000-3000Hz)
    freqs, psd = welch(signal, fs=fs, nperseg=min(1024, len(signal)//4))
    band_1 = np.sum(psd[(freqs >= 0) & (freqs < 200)])
    band_2 = np.sum(psd[(freqs >= 200) & (freqs < 1000)])
    band_3 = np.sum(psd[(freqs >= 1000) & (freqs < 3000)])

    total_energy = band_1 + band_2 + band_3
    band_energy_0_200 = band_1 / total_energy if total_energy > 0 else 0
    band_energy_200_1000 = band_2 / total_energy if total_energy > 0 else 0
    band_energy_1000_3000 = band_3 / total_energy if total_energy > 0 else 0

    # FFT features
    fft_vals = np.fft.rfft(signal)
    fft_freqs = np.fft.rfftfreq(len(signal), 1/fs)
    fft_mag = np.abs(fft_vals)
    # Peak frequency (excluding DC)
    if len(fft_mag) > 1:
        pos_mask = fft_freqs > 0
        if np.any(pos_mask):
            peak_freq = fft_freqs[pos_mask][np.argmax(fft_mag[pos_mask])]
        else:
            peak_freq = 0
    else:
        peak_freq = 0
    # Spectral centroid
    if np.sum(fft_mag) > 0:
        spectral_centroid = np.sum(fft_freqs * fft_mag) / np.sum(fft_mag)
    else:
        spectral_centroid = 0
    # Spectral rolloff (85% energy)
    cumsum = np.cumsum(fft_mag)
    if cumsum[-1] > 0:
        rolloff_thresh = 0.85 * cumsum[-1]
        rolloff_idx = np.where(cumsum >= rolloff_thresh)[0]
        if len(rolloff_idx) > 0:
            spectral_rolloff = fft_freqs[rolloff_idx[0]]
        else:
            spectral_rolloff = fft_freqs[-1] if len(fft_freqs) > 0 else 0
    else:
        spectral_rolloff = 0

    # 6. STFT-based features
    nperseg_stft = min(256, len(signal)//4)
    if nperseg_stft < 8:
        nperseg_stft = 8
    f_stft, t_stft, Zxx = stft(signal, fs=fs, nperseg=nperseg_stft)
    stft_mag = np.abs(Zxx)
    # Power spectral density averaged over time
    power_psd = np.mean(stft_mag**2, axis=1)  # average over time -> PSD
    if np.sum(power_psd) > 0:
        power_psd_norm = power_psd / np.sum(power_psd)
        stft_spectral_entropy = -np.sum(power_psd_norm * np.log2(power_psd_norm + 1e-12))
    else:
        stft_spectral_entropy = 0

    # CWT features
    widths = np.arange(1, 65)
    try:
        wavelet = 'morl'
        coefficients, frequencies = pywt.cwt(signal, widths, wavelet, sampling_period=1/fs)
        cwt_mag = np.abs(coefficients)
        cwt_energy = np.sum(cwt_mag**2, axis=1)
        if np.sum(cwt_energy) > 0:
            cwt_energy_norm = cwt_energy / np.sum(cwt_energy)
            cwt_entropy = -np.sum(cwt_energy_norm * np.log2(cwt_energy_norm + 1e-12))
        else:
            cwt_entropy = 0
    except:
        cwt_entropy = 0

    # Impact count
    abs_signal = np.abs(signal)
    height_threshold = 3 * rms
    min_samples = max(1, int(fs * 0.001))
    try:
        peaks, _ = find_peaks(abs_signal, height=height_threshold, distance=min_samples)
        impact_count = len(peaks)
    except:
        impact_count = 0

    return {
        "RMS": rms,
        "Peak": peak,
        "CrestFactor": crest_factor,
        "Kurtosis": kurt,
        "Skewness": skewness,
        "ShapeFactor": shape_factor,
        "ImpulseFactor": impulse_factor,
        "SpectralEntropy": spectral_entropy,
        "BandEnergy_0_200Hz": band_energy_0_200,
        "BandEnergy_200_1000Hz": band_energy_200_1000,
        "BandEnergy_1000_3000Hz": band_energy_1000_3000,
        "FFT_PeakFreq": peak_freq,
        "FFT_SpectralCentroid": spectral_centroid,
        "FFT_SpectralRolloff": spectral_rolloff,
        "STFT_SpectralEntropy": stft_spectral_entropy,
        "CWT_Entropy": cwt_entropy,
        "ImpactCount": impact_count
    }

# --- Основная часть скрипта для генерации датасета ---
if __name__ == "__main__":
    # Параметры эксперимента
    flat_sizes = np.arange(0, 46, 5)  # 0, 5, 10, 15, ..., 45 мм
    speeds = [5, 10, 15, 20]          # м/с
    repeats = 50                      # Повторов для каждого сценария (увеличено для лучшей статистики)
    fs = 10000                        # Частота дискретизации (Гц)
    T = 3.0                           # Время симуляции (сек) - достаточно для нескольких оборотов колеса

    dataset_results = []

    total_experiments = len(flat_sizes) * len(speeds) * repeats
    print(f"Generating RailGuard Research Dataset ({total_experiments} samples)...")
    print(f"Flat sizes: {flat_sizes} mm")
    print(f"Speeds: {speeds} m/s")
    print(f"Repeats per condition: {repeats}")

    experiment_count = 0
    for size in flat_sizes:
        for speed in speeds:
            for rep in range(repeats):
                # Симуляция с разным случайным шумом и вариацией параметров для robustness
                # Вариация параметров ±10% для имитации реальных вариаций
                mw_var = 150.0 * np.random.uniform(0.9, 1.1)
                mb_var = 3000.0 * np.random.uniform(0.9, 1.1)
                ks_var = 2e6 * np.random.uniform(0.9, 1.1)
                cs_var = 5000 * np.random.uniform(0.9, 1.1)
                kh_var = 1.5e8 * np.random.uniform(0.9, 1.1)
                noise_lvl_var = np.random.uniform(0.05, 0.2)

                # Временно переопределяем глобальные параметры для этой итерации
                # В реальном приложении лучше передавать их как аргументы функции
                global mw, mb, ks, cs, kh
                mw, mb, ks, cs, kh = mw_var, mb_var, ks_var, cs_var, kh_var

                try:
                    t, signal, _, contact_states = simulate_railguard_wheel_flat(
                        flat_length_mm=size,
                        speed_mps=speed,
                        T=T,
                        fs=fs,
                        noise_lvl=noise_lvl_var
                    )

                    feats = extract_advanced_features(signal, fs=fs)

                    row = {
                        "FlatSize": size,
                        "Speed": speed,
                        "Repeat": rep,
                        **feats
                    }
                    dataset_results.append(row)

                except Exception as e:
                    print(f"Error in simulation (size={size}mm, speed={speed}m/s, rep={rep}): {e}")
                    continue
                finally:
                    # Восстанавливаем исходные параметры
                    mw, mb, ks, cs, kh = 0.7, 2.0, 2e6, 5000, 1.5e8

                experiment_count += 1
                if experiment_count % 100 == 0:
                    print(f"Completed {experiment_count}/{total_experiments} experiments")

    # Сохранение результатов
    if dataset_results:
        df = pd.DataFrame(dataset_results)
        df.to_csv(os.path.join(base_dir, "railguard_robust_dataset.csv"), index=False)
        print(f"\nRobust Dataset saved to {os.path.join(base_dir, 'railguard_robust_dataset.csv')}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Базовая статистика по датасету
        print("\nDataset Statistics:")
        print(df.groupby('FlatSize')[['RMS', 'Kurtosis', 'ImpulseFactor']].agg(['mean', 'std']).round(4))

    else:
        print("No data generated due to errors!")

    # Пример визуализации (опционально, можно закомментировать для ускорения)
    try:
        if len(dataset_results) > 0:
            # Визуализация: Распределение Kurtosis для разного размера Flat
            plt.figure(figsize=(12, 8))

            # Subplot 1: Распределение Kurtosis
            plt.subplot(2, 2, 1)
            for size in flat_sizes[::2]:  # Показываем каждую вторую размер для наглядности
                if size in df['FlatSize'].values:
                    data = df[df['FlatSize'] == size]['Kurtosis']
                    plt.hist(data, bins=20, alpha=0.7, label=f'Flat {size}mm', density=True)
            plt.title("Kurtosis Distribution by Flat Size")
            plt.xlabel("Kurtosis Value")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Subplot 2: Среднее значение Kurtosis с ошибками
            plt.subplot(2, 2, 2)
            flat_sizes_present = sorted(df['FlatSize'].unique())
            kurt_means = []
            kurt_stds = []
            for size in flat_sizes_present:
                data = df[df['FlatSize'] == size]['Kurtosis']
                kurt_means.append(data.mean())
                kurt_stds.append(data.std())

            plt.errorbar(flat_sizes_present, kurt_means, yerr=kurt_stds,
                        fmt='o-', capsize=5, capthick=2)
            plt.title("Mean Kurtosis vs Flat Size (with Std Dev)")
            plt.xlabel("Flat Size (mm)")
            plt.ylabel("Kurtosis")
            plt.grid(True, alpha=0.3)

            # Subplot 3: Impulse Factor
            plt.subplot(2, 2, 3)
            impulse_means = []
            impulse_stds = []
            for size in flat_sizes_present:
                data = df[df['FlatSize'] == size]['ImpulseFactor']
                impulse_means.append(data.mean())
                impulse_stds.append(data.std())

            plt.errorbar(flat_sizes_present, impulse_means, yerr=impulse_stds,
                        fmt='s-', capsize=5, capthick=2, color='orange')
            plt.title("Mean Impulse Factor vs Flat Size")
            plt.xlabel("Flat Size (mm)")
            plt.ylabel("Impulse Factor")
            plt.grid(True, alpha=0.3)

            # Subplot 4: Band Energy Ratio
            plt.subplot(2, 2, 4)
            band_energy_means = []
            band_energy_stds = []
            for size in flat_sizes_present:
                data = df[df['FlatSize'] == size]['BandEnergy_1000_3000Hz']
                band_energy_means.append(data.mean())
                band_energy_stds.append(data.std())

            plt.errorbar(flat_sizes_present, band_energy_means, yerr=band_energy_stds,
                        fmt='^-', capsize=5, capthick=2, color='green')
            plt.title("Energy in 1000-3000 Hz Band vs Flat Size")
            plt.xlabel("Flat Size (mm)")
            plt.ylabel("Normalized Energy")
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            
            plt.savefig(os.path.join(base_dir, 'railguard_feature_analysis.png'), dpi=150, bbox_inches='tight')
            print(f"Feature analysis plots saved to {os.path.join(base_dir, 'railguard_feature_analysis.png')}")
            plt.show()  # Раскомментировать если хотите показать графики сразу
    except Exception as e:
        print(f"Could not generate plots: {e}")

print("\nGeneration complete!")