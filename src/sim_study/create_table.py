import json
import numpy as np

def format_row(method, metric, values, extra=None):
	row = f"{method} & {metric} "
	for v in values:
		if isinstance(v, float):
			row += f"& {v:.3f} "
		else:
			row += f"& {v} "
	if extra:
		row += extra
	row += r"\\"
	return row

def main():
	stats = json.load(open("results/sim_study/performance_stats.json", "r"))
	nf = stats['nf_stats']
	ncl = stats['ncl_stats']
	abc = stats['abc_stats']
	
	nf_cov = stats['nf_coverage']
	nf_ci = stats['nf_interval_lengths_mean']
	
	ncl_cov_G = stats['ncl_coverage_G']
	ncl_ci_G = stats['ncl_interval_lengths_G_mean']
	ncl_cov_platt = stats['ncl_coverage_platt']
	ncl_ci_platt = stats['ncl_interval_lengths_platt_mean']
	ncl_cov = stats['ncl_coverage_hessian']
	ncl_ci = stats['ncl_interval_lengths_hessian_mean']
	
	abc_cov = stats['abc_coverage']
	abc_ci = stats['abc_interval_lengths_mean']

	# Parameter names (edit as needed)
	param_names = [r"$\ln(\delta)$", r"$\ln(\sigma^2)$", r"$\beta_{0}$", r"$\beta_{1}$", r"$\beta_{2}$"]

	# Start LaTeX table
	lines = []
	lines.append(r"\begin{tabular}{c|c|ccccc}")
	lines.append(r"\hline")
	lines.append(r"Method & Metric " + " ".join([f"& {p}" for p in param_names]) + r"\\")
	lines.append(r"\hline")

	# NCL, Uncalibrated
	lines.append(format_row("NCL,", "Bias", ncl['bias']))
	mse_with_se_ncl = [f"{mse:.3f} ({se:.3f})" for mse, se in zip(ncl['rmse_paramwise'], ncl['rmse_paramwise_error'])]
	lines.append(format_row("Uncal.", "RMSE", mse_with_se_ncl))
	lines.append(format_row("", "Coverage", ncl_cov))
	lines.append(format_row("", "CI Len.", ncl_ci))
	lines.append(r"\hline")

	# NCL, Platt
	lines.append(format_row("NCL,", "Coverage", ncl_cov_platt))
	lines.append(format_row("Platt", "CI Len.", ncl_ci_platt))
	lines.append(r"\hline")

	# NCL, Godambe
	lines.append(format_row("NCL,", "Coverage", ncl_cov_G))
	lines.append(format_row("Godambe", "CI Len.", ncl_ci_G))
	lines.append(r"\hline")

	# NF
	lines.append(format_row("NF", "Bias", nf['bias']))
	mse_with_se = [f"{mse:.3f} ({se:.3f})" for mse, se in zip(nf['rmse_paramwise'], nf['rmse_paramwise_error'])]
	lines.append(format_row("", "RMSE", mse_with_se))
	lines.append(format_row("", "Coverage", nf_cov))
	lines.append(format_row("", "CI Len.", nf_ci))
	lines.append(r"\hline")

	# ABC
	lines.append(format_row("ABC", "Bias", abc['bias']))
	mse_with_se_abc = [f"{mse:.3f} ({se:.3f})" for mse, se in zip(abc['rmse_paramwise'], abc['rmse_paramwise_error'])]
	lines.append(format_row("", "RMSE", mse_with_se_abc))
	lines.append(format_row("", "Coverage", abc_cov))
	lines.append(format_row("", "CI Len.", abc_ci))
	lines.append(r"\hline")
	lines.append(r"\end{tabular}")

	# Write to file
	with open("results/sim_study/performance_table.tex", "w") as f:
		for line in lines:
			f.write(line + "\n")

if __name__ == "__main__":
	main()
