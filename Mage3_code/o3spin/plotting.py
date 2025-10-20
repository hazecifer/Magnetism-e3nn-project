import csv

def save_loss_curves(train_losses, val_losses, out_png, out_csv=None, title="Loss", xlabel="epoch", ylabel="loss"):
    """
    保存训练/验证损失曲线到 PNG（若 matplotlib 可用），并可选导出 CSV。
    - train_losses, val_losses: list[float]
    - out_png: 输出图片路径
    - out_csv: 输出 CSV 路径（可为 None）
    - 失败时自动降级为仅 CSV，并打印提示
    """
    # 保存 CSV
    if out_csv:
        try:
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "train_loss", "val_loss"])
                for i, (tr, va) in enumerate(zip(train_losses, val_losses), start=1):
                    w.writerow([i, float(tr), float(va)])
            print(f"已保存损失 CSV: {out_csv}")
        except Exception as e:
            print(f"保存 CSV 失败: {e}")

    # 绘图（如缺少 matplotlib 会自动跳过）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = list(range(1, len(train_losses) + 1))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, train_losses, label="train")
        ax.plot(xs, val_losses, label="val")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"已保存损失曲线: {out_png}")
    except Exception as e:
        print(f"Matplotlib 不可用或绘图失败（{e}）。已跳过绘图。")
