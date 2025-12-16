"""
Automated report generation for calcium imaging statistical analyses.

Generate publication-ready summaries with minimal effort. These tools help
you create comprehensive reports that include all the details needed for
reproducibility and proper interpretation.

Classes
-------
StatsReport : Container for building comprehensive statistical reports

Functions
---------
quick_report : One-line report generation
generate_methods_text : Create publication-ready methods paragraph
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import StatsResult

if TYPE_CHECKING:
    from wizards_staff.wizards.orb import Orb


class StatsReport:
    """
    Container for building comprehensive statistical reports.
    
    This class helps you build a complete analysis report step by step,
    then export it to various formats.
    
    Attributes
    ----------
    title : str
        Report title
    author : str
        Report author
    date : str
        Report date
    sections : list
        List of report sections
    
    Examples
    --------
    >>> report = StatsReport(title="Treatment Effects on Neural Activity")
    >>> report.add_data_summary(df, description="Sample firing rates")
    >>> report.add_comparison(frpm_result, section_title="Firing Rate Analysis")
    >>> report.generate("my_report.html")
    """
    
    def __init__(
        self,
        title: str,
        author: Optional[str] = None,
        date: Optional[str] = None
    ):
        """
        Initialize report with metadata.
        
        Parameters
        ----------
        title : str
            Report title.
        author : str, optional
            Report author name.
        date : str, optional
            Report date. Use "auto" for current date.
        """
        self.title = title
        self.author = author or "Wizards Staff"
        self.date = (
            datetime.now().strftime("%Y-%m-%d")
            if date == "auto" or date is None
            else date
        )
        self.sections = []
        self.figures = []
        self._methods_used = set()
        self._sample_sizes = {}
    
    def add_data_summary(
        self,
        data: pd.DataFrame,
        description: Optional[str] = None
    ) -> None:
        """
        Add summary statistics table for the dataset.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame to summarize.
        description : str, optional
            Description of the data.
        """
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Calculate summary statistics
        summary = data[numeric_cols].describe()
        
        # Add section
        self.sections.append({
            "type": "data_summary",
            "title": "Data Summary",
            "description": description,
            "n_samples": len(data),
            "n_columns": len(data.columns),
            "numeric_columns": list(numeric_cols),
            "summary_stats": summary.to_dict(),
            "sample_data": data.head(5).to_dict(),
        })
    
    def add_comparison(
        self,
        result: StatsResult,
        section_title: Optional[str] = None
    ) -> None:
        """
        Add a statistical comparison to the report.
        
        Parameters
        ----------
        result : StatsResult
            Result from a statistical test.
        section_title : str, optional
            Title for this section.
        """
        self._methods_used.add(result.test_name)
        self._sample_sizes.update(result.sample_sizes)
        
        self.sections.append({
            "type": "comparison",
            "title": section_title or f"{result.test_name} Results",
            "result": result,
        })
    
    def add_figure(
        self,
        fig: plt.Figure,
        caption: str,
        figure_id: Optional[str] = None
    ) -> None:
        """
        Add a figure to the report.
        
        Parameters
        ----------
        fig : plt.Figure
            Matplotlib figure to include.
        caption : str
            Figure caption.
        figure_id : str, optional
            Unique identifier for the figure.
        """
        fig_id = figure_id or f"fig_{len(self.figures) + 1}"
        self.figures.append({
            "id": fig_id,
            "figure": fig,
            "caption": caption,
        })
    
    def add_methods_text(
        self,
        tests_used: Optional[List[str]] = None,
        corrections_applied: Optional[str] = None,
        software_versions: bool = True
    ) -> str:
        """
        Generate methods section text suitable for publications.
        
        Parameters
        ----------
        tests_used : list of str, optional
            List of statistical tests used. Auto-detected if None.
        corrections_applied : str, optional
            Multiple comparison correction method used.
        software_versions : bool
            Whether to include software version information.
        
        Returns
        -------
        str
            Methods section text.
        """
        if tests_used is None:
            tests_used = list(self._methods_used)
        
        methods = generate_methods_text(
            analyses_performed=tests_used,
            sample_sizes=self._sample_sizes,
            corrections_applied=corrections_applied,
            software_versions=software_versions,
        )
        
        self.sections.append({
            "type": "methods",
            "title": "Statistical Methods",
            "content": methods,
        })
        
        return methods
    
    def generate(
        self,
        filepath: str,
        format: Optional[str] = None
    ) -> str:
        """
        Generate the final report file.
        
        Parameters
        ----------
        filepath : str
            Output file path.
        format : str, optional
            Output format. If None, inferred from filepath extension.
            Supported: "html", "markdown", "md"
        
        Returns
        -------
        str
            Path to the generated file.
        """
        if format is None:
            if filepath.endswith('.html'):
                format = "html"
            elif filepath.endswith('.md'):
                format = "markdown"
            else:
                format = "html"
        
        if format == "html":
            content = self._generate_html()
        elif format in ("markdown", "md"):
            content = self._generate_markdown()
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return filepath
    
    def _generate_html(self) -> str:
        """Generate HTML report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
        }}
        .meta {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .result-box {{
            background: #ecf0f1;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
        }}
        .interpretation {{
            background: #e8f6f3;
            border-left: 4px solid #1abc9c;
            padding: 15px;
            margin: 15px 0;
        }}
        .warning {{
            background: #fef9e7;
            border-left: 4px solid #f39c12;
            padding: 15px;
            margin: 15px 0;
        }}
        .significant {{
            color: #27ae60;
            font-weight: bold;
        }}
        .not-significant {{
            color: #e74c3c;
        }}
        pre {{
            background: #f4f4f4;
            padding: 10px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <div class="meta">
        <p>Author: {self.author}</p>
        <p>Date: {self.date}</p>
        <p>Generated by Wizards Staff Statistical Analysis Module</p>
    </div>
"""
        
        for section in self.sections:
            if section["type"] == "data_summary":
                html += self._html_data_summary(section)
            elif section["type"] == "comparison":
                html += self._html_comparison(section)
            elif section["type"] == "methods":
                html += self._html_methods(section)
        
        html += """
</body>
</html>"""
        
        return html
    
    def _html_data_summary(self, section: dict) -> str:
        """Generate HTML for data summary section."""
        html = f"""
    <h2>{section['title']}</h2>
    <p>{section.get('description', '')}</p>
    <p><strong>Dataset:</strong> {section['n_samples']} samples, {section['n_columns']} variables</p>
    
    <h3>Summary Statistics</h3>
    <table>
        <tr><th>Statistic</th>"""
        
        # Add column headers
        for col in section["numeric_columns"][:5]:  # Limit columns
            html += f"<th>{col}</th>"
        html += "</tr>\n"
        
        # Add statistics rows
        stats = section["summary_stats"]
        for stat in ["count", "mean", "std", "min", "max"]:
            html += f"        <tr><td>{stat}</td>"
            for col in section["numeric_columns"][:5]:
                val = stats.get(col, {}).get(stat, "N/A")
                if isinstance(val, float):
                    html += f"<td>{val:.3f}</td>"
                else:
                    html += f"<td>{val}</td>"
            html += "</tr>\n"
        
        html += "    </table>\n"
        return html
    
    def _html_comparison(self, section: dict) -> str:
        """Generate HTML for comparison section."""
        result = section["result"]
        sig_class = "significant" if result.p_value < 0.05 else "not-significant"
        
        html = f"""
    <h2>{section['title']}</h2>
    
    <div class="result-box">
        <p><strong>Test:</strong> {result.test_name}</p>
        <p><strong>Statistic:</strong> {result.statistic:.4f}</p>
        <p><strong>P-value:</strong> <span class="{sig_class}">{result.p_value:.4f}</span></p>
"""
        
        if result.effect_size_type:
            html += f"""        <p><strong>Effect Size ({result.effect_size_type}):</strong> {result.effect_size:.3f} ({result.effect_size_magnitude})</p>
"""
        
        if result.ci_lower is not None:
            html += f"""        <p><strong>95% CI:</strong> [{result.ci_lower:.3f}, {result.ci_upper:.3f}]</p>
"""
        
        html += "    </div>\n"
        
        # Group statistics
        if result.group_stats:
            html += """    <h3>Group Statistics</h3>
    <table>
        <tr><th>Group</th><th>N</th><th>Mean</th><th>Median</th><th>Std</th></tr>
"""
            for group, stats in result.group_stats.items():
                n = result.sample_sizes.get(group, "N/A")
                html += f"""        <tr>
            <td>{group}</td>
            <td>{n}</td>
            <td>{stats.get('mean', 0):.3f}</td>
            <td>{stats.get('median', 0):.3f}</td>
            <td>{stats.get('std', 0):.3f}</td>
        </tr>
"""
            html += "    </table>\n"
        
        # Interpretation
        if result.interpretation:
            html += f"""
    <div class="interpretation">
        <strong>Interpretation:</strong><br>
        {result.interpretation}
    </div>
"""
        
        # Warnings
        if result.warnings:
            html += """    <div class="warning">
        <strong>⚠️ Warnings:</strong><br>
"""
            for warning in result.warnings:
                html += f"        • {warning}<br>\n"
            html += "    </div>\n"
        
        # Post-hoc table
        if result.post_hoc_table is not None:
            html += """    <h3>Post-hoc Comparisons</h3>
    <table>
        <tr>"""
            for col in result.post_hoc_table.columns:
                html += f"<th>{col}</th>"
            html += "</tr>\n"
            
            for _, row in result.post_hoc_table.iterrows():
                html += "        <tr>"
                for col in result.post_hoc_table.columns:
                    val = row[col]
                    if isinstance(val, float):
                        html += f"<td>{val:.4f}</td>"
                    elif isinstance(val, bool):
                        html += f"<td>{'Yes' if val else 'No'}</td>"
                    else:
                        html += f"<td>{val}</td>"
                html += "</tr>\n"
            html += "    </table>\n"
        
        return html
    
    def _html_methods(self, section: dict) -> str:
        """Generate HTML for methods section."""
        return f"""
    <h2>{section['title']}</h2>
    <div class="result-box">
        <p>{section['content']}</p>
    </div>
"""
    
    def _generate_markdown(self) -> str:
        """Generate Markdown report."""
        md = f"""# {self.title}

**Author:** {self.author}  
**Date:** {self.date}  
**Generated by:** Wizards Staff Statistical Analysis Module

---

"""
        
        for section in self.sections:
            if section["type"] == "data_summary":
                md += self._md_data_summary(section)
            elif section["type"] == "comparison":
                md += self._md_comparison(section)
            elif section["type"] == "methods":
                md += f"## {section['title']}\n\n{section['content']}\n\n"
        
        return md
    
    def _md_data_summary(self, section: dict) -> str:
        """Generate Markdown for data summary."""
        md = f"## {section['title']}\n\n"
        if section.get('description'):
            md += f"{section['description']}\n\n"
        md += f"**Dataset:** {section['n_samples']} samples, {section['n_columns']} variables\n\n"
        return md
    
    def _md_comparison(self, section: dict) -> str:
        """Generate Markdown for comparison."""
        result = section["result"]
        md = f"## {section['title']}\n\n"
        md += f"**Test:** {result.test_name}\n\n"
        md += f"| Metric | Value |\n|--------|-------|\n"
        md += f"| Statistic | {result.statistic:.4f} |\n"
        md += f"| P-value | {result.p_value:.4f} |\n"
        if result.effect_size_type:
            md += f"| Effect Size ({result.effect_size_type}) | {result.effect_size:.3f} ({result.effect_size_magnitude}) |\n"
        md += "\n"
        
        if result.interpretation:
            md += f"**Interpretation:** {result.interpretation}\n\n"
        
        return md


def quick_report(
    orb: "Orb",
    group_col: str,
    metrics: List[str] = None,
    output_path: str = "analysis_report.html"
) -> str:
    """
    One-line function to generate a complete analysis report.
    
    Analyzes all specified metrics, compares between groups, and generates
    a comprehensive HTML report.
    
    Parameters
    ----------
    orb : Orb
        Wizards Staff Orb object with completed analysis.
    group_col : str
        Column in metadata defining groups to compare.
    metrics : list of str, optional
        Metrics to analyze. Default is ["frpm", "fwhm", "rise_time"].
    output_path : str
        Path for the output report file.
    
    Returns
    -------
    str
        Path to the generated report.
    
    Examples
    --------
    >>> quick_report(orb, group_col="Treatment", output_path="results.html")
    """
    from .core import prepare_for_stats
    from .tests import compare_two_groups
    
    if metrics is None:
        metrics = ["frpm", "fwhm", "rise_time"]
    
    # Initialize report
    report = StatsReport(
        title=f"Calcium Imaging Statistical Analysis: {group_col} Comparison",
        date="auto"
    )
    
    # Analyze each metric
    for metric in metrics:
        try:
            df = prepare_for_stats(orb, metric=metric, metadata_cols=[group_col])
            
            # Find the aggregated metric column
            metric_cols = [c for c in df.columns if c.startswith("mean_")]
            if not metric_cols:
                continue
            
            metric_col = metric_cols[0]
            
            # Add data summary
            report.add_data_summary(df, description=f"{metric.upper()} data summary")
            
            # Run comparison
            result = compare_two_groups(
                data=df,
                group_col=group_col,
                metric_col=metric_col
            )
            
            report.add_comparison(result, section_title=f"{metric.upper()} Comparison")
            
        except Exception as e:
            print(f"Warning: Could not analyze {metric}: {e}")
    
    # Add methods section
    report.add_methods_text(software_versions=True)
    
    # Generate report
    report.generate(output_path)
    
    return output_path


def generate_methods_text(
    analyses_performed: List[str],
    sample_sizes: Dict[str, int] = None,
    corrections_applied: Optional[str] = None,
    software_versions: bool = True
) -> str:
    """
    Generate publication-ready methods paragraph.
    
    Parameters
    ----------
    analyses_performed : list of str
        Names of statistical tests used.
    sample_sizes : dict, optional
        Sample sizes per group.
    corrections_applied : str, optional
        Multiple comparison correction method.
    software_versions : bool
        Whether to include software version information.
    
    Returns
    -------
    str
        Methods paragraph suitable for publication.
    
    Examples
    --------
    >>> text = generate_methods_text(
    ...     analyses_performed=["Mann-Whitney U test", "Kruskal-Wallis"],
    ...     sample_sizes={"Control": 10, "Treatment": 12},
    ...     corrections_applied="Benjamini-Hochberg FDR"
    ... )
    >>> print(text)
    """
    parts = []
    
    # Software information
    if software_versions:
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        parts.append(
            f"Statistical analyses were performed using Wizards Staff "
            f"(https://github.com/ArcInstitute/Wizards-Staff) with Python {python_version}."
        )
    
    # Sample sizes
    if sample_sizes:
        sizes = ", ".join([f"{g}: n={n}" for g, n in sample_sizes.items()])
        parts.append(f"Group sample sizes were: {sizes}.")
    
    # Tests used
    if analyses_performed:
        if len(analyses_performed) == 1:
            parts.append(f"Comparisons were performed using {analyses_performed[0]}.")
        else:
            tests = ", ".join(analyses_performed[:-1]) + f" and {analyses_performed[-1]}"
            parts.append(f"Statistical tests included {tests}.")
    
    # Corrections
    if corrections_applied:
        parts.append(
            f"Multiple comparisons were corrected using {corrections_applied} "
            f"to control for false discoveries."
        )
    
    # Effect sizes
    parts.append(
        "Effect sizes are reported alongside p-values to assess practical significance."
    )
    
    # Significance level
    parts.append("Statistical significance was set at α = 0.05.")
    
    return " ".join(parts)

