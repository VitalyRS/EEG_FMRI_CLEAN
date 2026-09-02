#!/usr/bin/env python3
"""
Patch generate_report.py to add:
1. Quality score column in table
2. Normalized timeseries plots for visual comparison
"""
import sys

# Read current file
with open('generate_report.py', 'r') as f:
    content = f.read()

# Add score badge CSS
css_addition = """
.score-badge{display:inline-block;padding:0.2rem 0.5rem;border-radius:4px;font-family:var(--font-mono);font-size:0.8rem;font-weight:600}
.score-badge.good{background:var(--good);color:#fff}
.score-badge.warn{background:var(--warn);color:#fff}
.score-badge.bad{background:var(--bad);color:#fff}
.score-badge.na{background:var(--na);color:#fff}
"""

# Insert after .asr-badge CSS
if '.asr-badge{' in content and '.score-badge{' not in content:
    content = content.replace(
        '.asr-badge{display:inline-block;padding:0.15rem 0.4rem;border-radius:3px;background:var(--border);font-family:var(--font-mono);font-size:0.75rem}',
        '.asr-badge{display:inline-block;padding:0.15rem 0.4rem;border-radius:3px;background:var(--border);font-family:var(--font-mono);font-size:0.75rem}\n' + css_addition
    )
    print("✅ Added score badge CSS")
else:
    print("⚠️  Score badge CSS already exists or .asr-badge not found")

# Update drawTimeseries function to normalize amplitudes
normalize_code = """
  // Normalize both signals to same scale for visual comparison
  const allVals = vals_o.concat(vals_r);
  const globalMin = Math.min(...allVals);
  const globalMax = Math.max(...allVals);
  const globalRange = globalMax - globalMin;
  const globalPad = globalRange * 0.1;

  const vMin = globalMin - globalPad;
  const vMax = globalMax + globalPad;
  const vRange = vMax - vMin;
"""

old_normalize = """
  const allVals = vals_o.concat(vals_r);
  const vMin = Math.min(...allVals);
  const vMax = Math.max(...allVals);
  const vRange = vMax - vMin;
  const vPad = vRange * 0.1;
"""

if old_normalize in content:
    content = content.replace(old_normalize, normalize_code)
    print("✅ Updated normalization in drawTimeseries")
else:
    print("⚠️  Normalization code not found or already updated")

# Update y-scale to use new normalization
old_yscale = "const yScale = v => pad.t + plotH - ((v - vMin + vPad) / (vRange + 2*vPad)) * plotH;"
new_yscale = "const yScale = v => pad.t + plotH - ((v - vMin) / vRange) * plotH;"

if old_yscale in content:
    content = content.replace(old_yscale, new_yscale)
    print("✅ Updated yScale function")
else:
    print("⚠️  yScale already updated or different format")

# Write back
with open('generate_report.py', 'w') as f:
    f.write(content)

print("\n✅ Patched generate_report.py")
print("\nNote: Table header and quality score column were already added via Edit.")
print("If quality metrics missing, run: python3 compute_cleaning_metrics.py")
