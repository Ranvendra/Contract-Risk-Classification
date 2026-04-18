import io
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        # Logo/Title
        self.set_font("helvetica", "B", 18)
        self.set_text_color(15, 23, 42)  # slate-900
        self.cell(0, 10, "Legal Risk Analysis Report", border=0, align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_font("helvetica", "", 10)
        self.set_text_color(100, 116, 139) # slate-500
        self.cell(0, 6, "CONFIDENTIAL & PRIVILEGED", border=0, align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)
        # Main Line
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

def _safe_str(t) -> str:
    if t is None:
        return ""
    # Convert dash
    t = str(t).replace("—", "-")
    # latin-1 replace
    return t.encode('latin-1', 'replace').decode('latin-1')

def generate_pdf_report(results, domain: str, file_name: str = "Unknown Document") -> bytes:
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Overview
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, f"Document: {_safe_str(file_name)}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 8, f"Domain Context: {_safe_str(domain)}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(71, 85, 105)
    pdf.multi_cell(0, 6, f"This document has been evaluated against standard practices for {_safe_str(domain)} agreements. {len(results)} clauses were flagged for review.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    for i, cur in enumerate(results, 1):
        an = cur.get("analysis", {})
        risk = cur.get("risk_level", "Unknown")
        
        # Risk Badge colors
        if risk == "High":
            r, g, b = 220, 38, 38 # red-600
        elif risk == "Medium":
            r, g, b = 234, 88, 12 # orange-600
        else:
            r, g, b = 22, 163, 74 # green-600

        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(r, g, b)
        pdf.cell(0, 8, f"{i}. Clause Analysis - {_safe_str(risk)} Risk", border="B", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("helvetica", "B", 9)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(30, 8, "Risk Bearer: ")
        pdf.set_font("helvetica", "", 9)
        pdf.cell(60, 8, _safe_str(an.get('who_bears_the_risk', 'Unknown')))
        
        pdf.set_font("helvetica", "B", 9)
        pdf.cell(20, 8, "Action: ")
        pdf.set_font("helvetica", "", 9)
        pdf.cell(60, 8, _safe_str(an.get('action_required', 'Review')), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # What it Says
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(37, 99, 235)  # blue-600
        pdf.cell(0, 6, "What It Says", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(51, 65, 85) # slate-700
        pdf.multi_cell(0, 6, _safe_str(an.get('plain_english_summary', '-')), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        if risk in ("High", "Medium"):
            # Why Risk
            pdf.set_font("helvetica", "B", 10)
            pdf.set_text_color(220, 38, 38)  # red-600
            pdf.cell(0, 6, "Why It's Risky", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "", 10)
            pdf.set_text_color(51, 65, 85)
            pdf.multi_cell(0, 6, _safe_str(an.get('what_makes_it_risky', '-')), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

            # Domain Practice
            pdf.set_font("helvetica", "B", 10)
            pdf.set_text_color(13, 148, 136)  # emerald-600
            pdf.cell(0, 6, f"{_safe_str(domain)} Practice", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "", 10)
            pdf.set_text_color(51, 65, 85)
            pdf.multi_cell(0, 6, _safe_str(an.get('industry_standard_practice', '-')), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

            # Negotiation
            pdf.set_font("helvetica", "B", 10)
            pdf.set_text_color(124, 58, 237)  # violet-600
            pdf.cell(0, 6, "Negotiation & Mitigation", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "", 10)
            pdf.set_text_color(51, 65, 85)
            pdf.multi_cell(0, 6, _safe_str(an.get('negotiation_tips', '-')), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

            # Rewrite
            pdf.set_font("helvetica", "B", 10)
            pdf.set_text_color(100, 116, 139)  # slate-500
            pdf.cell(0, 6, "Safer Rewrite", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("courier", "", 9)
            pdf.set_text_color(71, 85, 105)
            pdf.set_fill_color(248, 250, 252) # slate-50
            pdf.multi_cell(0, 5, _safe_str(an.get('safer_rewrite', '-')), fill=True, border=1, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(8)
        else:
            pdf.set_font("helvetica", "B", 10)
            pdf.set_text_color(22, 163, 74)
            pdf.cell(0, 6, "Assessment", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "", 10)
            pdf.set_text_color(51, 65, 85)
            pdf.multi_cell(0, 6, "This clause is standard and lower risk. No immediate modifications required.", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(8)

    return bytes(pdf.output())

