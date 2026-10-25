"""
Execute Candy_Crush_Master_EDA.ipynb and embed all text outputs and
high-resolution Matplotlib figures into cell['outputs'].
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

def execute_and_embed(input_path, output_path):
    with open(input_path, 'r') as f:
        nb = json.load(f)

    global_scope = {
        '__file__': os.path.abspath(input_path)
    }

    execution_count = 1

    for cell_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        source = ''.join(cell['source'])
        print(f"Executing cell {cell_idx + 1}/{len(nb['cells'])}...")

        cell['execution_count'] = execution_count
        cell['outputs'] = []

        # Capture stdout
        stdout_capture = io.StringIO()
        plt.close('all')

        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(source, global_scope)

            # Check for printed text
            printed_text = stdout_capture.getvalue()
            if printed_text:
                cell['outputs'].append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": printed_text.splitlines(keepends=True)
                })

            # Check for generated Matplotlib figures
            figs = [plt.figure(num) for num in plt.get_fignums()]
            for fig in figs:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                cell['outputs'].append({
                    "data": {
                        "image/png": img_base64,
                        "text/plain": [f"<Figure size {fig.get_size_inches()[0]*100}x{fig.get_size_inches()[1]*100} with {len(fig.axes)} Axes>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
                plt.close(fig)

            execution_count += 1
            print(f"Cell {cell_idx + 1} succeeded (outputs: {len(cell['outputs'])}).")

        except Exception as e:
            print(f"Cell {cell_idx + 1} failed with error: {e}")
            cell['outputs'].append({
                "ename": type(e).__name__,
                "evalue": str(e),
                "output_type": "error",
                "traceback": [f"{type(e).__name__}: {str(e)}"]
            })
            raise e

    with open(output_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully saved executed notebook to {output_path}!")

if __name__ == '__main__':
    execute_and_embed('notebooks/Candy_Crush_Master_EDA.ipynb', 'notebooks/Candy_Crush_Master_EDA.ipynb')
    execute_and_embed('notebooks/Candy_Crush_Master_EDA.ipynb', 'kaggle_deploy/eda_kernel/Candy_Crush_Master_EDA.ipynb')
