import numpy as np
import matplotlib.pyplot as plt

DEG2RAD = np.pi / 180.0


def generate_case_A1(
    num_cycles=500,
    fs=3840.0,
    f0_nom=60.0,
    harmonics=(1, 3, 5, 7),
    freq_dev=0.005,   # ±0.5%
    amp_dev=0.01,     # ±1%
    snr_db=26.0,
    rng=None,
):
    """
    生成 Case A-1 数据：
    - 基波 60Hz，频率偏差 ±0.5%
    - 1/3/5/7 次谐波幅值分别为 100, 30, 20, 15，幅度抖动 ±1%
    - 相位分别为 152°, 35°, 0°, 0°
    - 加 26 dB 高斯白噪声

    返回:
        signals: (num_cycles, samples_per_cycle)
        labels:  (num_cycles, len(harmonics))
    """
    if rng is None:
        rng = np.random.default_rng()

    samples_per_cycle = int(round(fs / f0_nom))  # 64
    t = np.arange(samples_per_cycle) / fs

    # 基准幅值（你给的）
    base_amps = {
        1: 100.0,
        3: 30.0,
        5: 20.0,
        7: 15.0,
    }

    # 基准相位（度 → 弧度）
    base_phases = {
        1: 152.0 * DEG2RAD,
        3: 35.0 * DEG2RAD,
        5: 0.0 * DEG2RAD,
        7: 0.0 * DEG2RAD,
    }

    signals = np.zeros((num_cycles, samples_per_cycle), dtype=np.float64)
    labels = np.zeros((num_cycles, len(harmonics)), dtype=np.float64)

    for n in range(num_cycles):
        # 随机工频偏差
        f0 = f0_nom * (1.0 + rng.uniform(-freq_dev, freq_dev))

        # 幅值抖动
        amps_this_cycle = {}
        for h in harmonics:
            A0 = base_amps[h]
            A = A0 * (1.0 + rng.uniform(-amp_dev, amp_dev))
            amps_this_cycle[h] = A

        # 合成谐波（无噪声）
        s = np.zeros_like(t, dtype=np.float64)
        for h in harmonics:
            A = amps_this_cycle[h]
            phi = base_phases[h]
            s += A * np.sin(2.0 * np.pi * (h * f0) * t + phi)

        # 加 26 dB 高斯白噪声
        sig_power = np.mean(s ** 2)
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=s.shape)
        s_noisy = s + noise

        signals[n, :] = s_noisy
        labels[n, :] = [amps_this_cycle[h] for h in harmonics]

    return signals, labels


def generate_case_A2(
    num_cycles=500,
    fs=3840.0,
    f0_nom=60.0,
    harmonics=(1, 3, 5, 7),
    freq_dev=0.005,
    amp_dev=0.01,
    snr_db=26.0,
    rng=None,
):
    """
    生成 Case A-2 数据：
    - 与 A-1 相同的 1/3/5/7 次谐波设置
    - 额外叠加两个互谐波：
        f = 84.4 Hz,  A = 10, phase = 0
        f = 385.6 Hz, A = 5,  phase = 0
    """
    if rng is None:
        rng = np.random.default_rng()

    samples_per_cycle = int(round(fs / f0_nom))
    t = np.arange(samples_per_cycle) / fs

    base_amps = {
        1: 100.0,
        3: 30.0,
        5: 20.0,
        7: 15.0,
    }
    base_phases = {
        1: 152.0 * DEG2RAD,
        3: 35.0 * DEG2RAD,
        5: 0.0 * DEG2RAD,
        7: 0.0 * DEG2RAD,
    }

    interharmonics = [
        {"f": 84.4,  "A": 10.0, "phi": 0.0},
        {"f": 385.6, "A": 5.0,  "phi": 0.0},
    ]

    signals = np.zeros((num_cycles, samples_per_cycle), dtype=np.float64)
    labels = np.zeros((num_cycles, len(harmonics)), dtype=np.float64)

    for n in range(num_cycles):
        # 基本谐波部分
        f0 = f0_nom * (1.0 + rng.uniform(-freq_dev, freq_dev))

        amps_this_cycle = {}
        for h in harmonics:
            A0 = base_amps[h]
            A = A0 * (1.0 + rng.uniform(-amp_dev, amp_dev))
            amps_this_cycle[h] = A

        s = np.zeros_like(t, dtype=np.float64)
        for h in harmonics:
            A = amps_this_cycle[h]
            phi = base_phases[h]
            s += A * np.sin(2.0 * np.pi * (h * f0) * t + phi)

        # 叠加互谐波
        for comp in interharmonics:
            f_ih = comp["f"]
            A_ih = comp["A"]
            phi_ih = comp["phi"]
            s += A_ih * np.sin(2.0 * np.pi * f_ih * t + phi_ih)

        # 加 26 dB 噪声
        sig_power = np.mean(s ** 2)
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=s.shape)
        s_noisy = s + noise

        signals[n, :] = s_noisy
        labels[n, :] = [amps_this_cycle[h] for h in harmonics]

    return signals, labels


def save_datasets():
    rng = np.random.default_rng(42)

    # 生成 A-1
    X_A1, y_A1 = generate_case_A1(num_cycles=500, rng=rng)
    np.savez("caseA1.npz", signals=X_A1, labels=y_A1)
    print("Saved caseA1.npz:", X_A1.shape, y_A1.shape)

    # 生成 A-2
    X_A2, y_A2 = generate_case_A2(num_cycles=500, rng=rng)
    np.savez("caseA2.npz", signals=X_A2, labels=y_A2)
    print("Saved caseA2.npz:", X_A2.shape, y_A2.shape)


def check_one_sample(filename="caseA1.npz", index=0):
    """
    从 npz 文件中读取数据，打印 shape，并画出第 index 条样本的波形。
    """
    data = np.load(filename)
    signals = data["signals"]
    labels = data["labels"]

    print(f"{filename} signals shape:", signals.shape)  # (N, 64)
    print(f"{filename} labels shape:", labels.shape)    # (N, 4)

    x = signals[index]
    print(f"One sample shape: {x.shape}")

    # 画一条波形
    plt.figure(figsize=(8, 3))
    plt.plot(x, marker="o")
    plt.title(f"{filename} sample #{index}")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 顺便打印一下对应的标签
    print(f"Label for sample #{index} (A1-> [A1,A3,A5,A7]):", labels[index])


if __name__ == "__main__":
    # 第一步：生成并保存 A1 / A2 数据
    save_datasets()

    # 第二步：检查一个样本的形状和波形（例如 A1 第 0 条）
    check_one_sample("caseA1.npz", index=0)

    # 你也可以改成检查 A2：
    # check_one_sample("caseA2.npz", index=0)
