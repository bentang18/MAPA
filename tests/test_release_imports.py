"""Release entry points import without reading result files or parsing application arguments."""
import subprocess
import sys


def test_figure_import_does_not_render_or_parse_cli(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import evals.figures.fig_r31_label_efficiency_log10", "--unrelated-arg"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stdout
