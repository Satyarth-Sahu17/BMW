import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

class ReportGenerator:
    def __init__(self, experiment_dir):
        self.exp_dir = Path(experiment_dir)
        self.metrics_path = self.exp_dir / "metrics.json"
        self.conf_matrix_path = self.exp_dir / "confusion_matrix.png"
        self.roc_path = self.exp_dir / "roc_curves.png"
        
    def generate_html_report(self, output_path=None):
        if not output_path:
            output_path = self.exp_dir / "report.html"
            
        with open(self.metrics_path) as f:
            metrics = json.load(f)
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BMW Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #2c3e50; }}
                .metric-box {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .img-container {{ display: flex; gap: 20px; flex-wrap: wrap; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>BioAcoustic Monitoring - Evaluation Report</h1>
            <p>Generated on: {timestamp}</p>
            
            <div class="metric-box">
                <h2>Summary Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Accuracy</td><td>{metrics.get('accuracy', 'N/A'):.4f}</td></tr>
                    <tr><td>F1 Macro</td><td>{metrics.get('f1_macro', 'N/A'):.4f}</td></tr>
                    <tr><td>Precision Macro</td><td>{metrics.get('precision_macro', 'N/A'):.4f}</td></tr>
                    <tr><td>Recall Macro</td><td>{metrics.get('recall_macro', 'N/A'):.4f}</td></tr>
                    <tr><td>MCC</td><td>{metrics.get('mcc', 'N/A'):.4f}</td></tr>
                </table>
            </div>
            
            <h2>Per-Class Performance</h2>
            {pd.DataFrame(metrics.get('per_class', {})).transpose().to_html()}
            
            <h2>Visualizations</h2>
            <div class="img-container">
                <div>
                    <h3>Confusion Matrix</h3>
                    <img src="confusion_matrix.png" alt="Confusion Matrix">
                </div>
                <div>
                    <h3>ROC Curves</h3>
                    <img src="roc_curves.png" alt="ROC Curves">
                </div>
            </div>
            
            <h2>Model Configuration</h2>
            <pre>
            {self._load_config()}
            </pre>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
        print(f"Report generated at {output_path}")
        
    def _load_config(self):
        # Try to find config file in parent dirs or exp dir
        config_files = list(self.exp_dir.glob("*.yaml"))
        if config_files:
            with open(config_files[0]) as f:
                return f.read()
        return "Config file not found in experiment directory."

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        gen = ReportGenerator(sys.argv[1])
        gen.generate_html_report()
