"""
Execute both new notebooks:
1. notebooks/AlphaCandy_Grandmaster_Interactive_Showcase.ipynb
2. notebooks/AlphaCandy_Kaggle_GPU_2500_Benchmark_Analysis.ipynb
capturing all stdout streams and Matplotlib figures directly into the JSON.
"""

import json
import io
import sys
import os
import base64
import contextlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_notebook(nb_path):
    print(f"\n=======================================================")
    print(f"  RUNNING NOTEBOOK: {nb_path}")
    print(f"=======================================================")
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    # Global execution namespace
    global_scope = {
        '__file__': os.path.abspath(nb_path),
        '__name__': '__main__'
    }

    execution_count = 1

    for cell_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        source = ''.join(cell['source'])
        print(f"Executing Code Cell {cell_idx + 1}/{len(nb['cells'])}...")

        cell['execution_count'] = execution_count
        cell['outputs'] = []

        stdout_capture = io.StringIO()
        plt.close('all')

        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(source, global_scope)

            # 1. Capture stdout
            text_out = stdout_capture.getvalue()
            if text_out:
                cell['outputs'].append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": text_out.splitlines(keepends=True)
                })

            # 2. Capture figures
            figs = [plt.figure(num) for num in plt.get_fignums()]
            for fig in figs:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                cell['outputs'].append({
                    "data": {
                        "image/png": img_b64,
                        "text/plain": [f"<Figure size {fig.get_size_inches()[0]*100}x{fig.get_size_inches()[1]*100} with {len(fig.axes)} Axes>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
                plt.close(fig)

            execution_count += 1
            print(f"  -> Cell {cell_idx + 1} Succeeded! ({len(cell['outputs'])} output items)")

        except Exception as e:
            print(f"  -> Cell {cell_idx + 1} FAILED: {e}")
            import traceback
            traceback.print_exc()
            cell['outputs'].append({
                "ename": type(e).__name__,
                "evalue": str(e),
                "output_type": "error",
                "traceback": traceback.format_exc().splitlines(keepends=True)
            })
            raise e

    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully finished and saved: {nb_path}!")

if __name__ == '__main__':
    # Run Notebook 2 first (data analysis)
    run_notebook('notebooks/AlphaCandy_Kaggle_GPU_2500_Benchmark_Analysis.ipynb')
    
    # Run Notebook 1 (showcase)
    run_notebook('notebooks/AlphaCandy_Grandmaster_Interactive_Showcase.ipynb')
