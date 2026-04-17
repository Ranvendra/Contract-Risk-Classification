#!/usr/bin/env python3
"""
rag_setup.py — Domain-Aware RAG Knowledge Base Initialization
=============================================================

Run ONCE before launching the app to build the vector store:

    python rag_setup.py

What this does:
  1. Creates a persistent ChromaDB database at  data/chroma_db/
  2. Embeds 35+ high-quality legal guidelines using sentence-transformers
  3. Tags every entry with { domain, topic } metadata for precision filtering

Supported domains:
  NDA  |  Employment  |  Lease  |  SaaS  |  Vendor  |  General
"""

from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
COLLECTION_NAME = "legal_guidelines"

# ─────────────────────────────────────────────────────────────────────────────
# HIGH-QUALITY DOMAIN-SPECIFIC LEGAL KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE: list[dict] = [

    # ──────────────────────────────────────────────────────────────────────────
    # DOMAIN: NDA
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "nda-conf-01",
        "domain": "NDA",
        "topic": "confidentiality",
        "title": "Defining Confidential Information Precisely",
        "text": (
            "A well-drafted NDA defines 'Confidential Information' narrowly and with standard exceptions. "
            "Best practice: limit the definition to information marked as confidential at disclosure, "
            "or disclosed in circumstances where confidentiality is reasonably understood. "
            "Mandatory standard exclusions that should always appear: (1) information already in the public "
            "domain through no fault of the receiving party, (2) information independently developed without "
            "reference to the disclosing party's information, and (3) information received lawfully from a "
            "third party without restriction. Overly broad definitions such as 'all information exchanged "
            "in any form or medium' are frequently found unenforceable by courts as they impose an "
            "unreasonable and indefinite restraint on the receiving party's use of their own knowledge."
        ),
    },
    {
        "id": "nda-scope-01",
        "domain": "NDA",
        "topic": "permitted_use",
        "title": "Permitted Use and Purpose Limitation",
        "text": (
            "An NDA should expressly state a specific 'Permitted Purpose' — the sole reason the receiving "
            "party may use the disclosed information. One-sided NDAs where the disclosing party retains "
            "full freedom to use the counterpart's information while restricting the reverse create "
            "dangerous asymmetric risk. Best practice: mutual NDAs with symmetrical permitted purposes, "
            "or a narrowly scoped unilateral NDA tied to a specific identified project "
            "(e.g., 'evaluation of a potential acquisition of Company X during Q3 2025'). "
            "Any use outside the declared permitted purpose should constitute material breach. "
            "Watch for ambiguous language such as 'internal business purposes' — this is dangerously "
            "broad and may permit competitive use of disclosed information."
        ),
    },
    {
        "id": "nda-duration-01",
        "domain": "NDA",
        "topic": "duration",
        "title": "Duration: Confidentiality Obligations and Survival",
        "text": (
            "Standard NDA confidentiality obligations survive 2–5 years after contract termination. "
            "Perpetual obligations ('remain confidential forever') are routinely deemed unenforceable "
            "by courts as they create an unreasonable and indefinite restraint on trade. "
            "Best practice: clearly distinguish between (1) regular confidential information — a 3–5 year "
            "post-termination obligation, and (2) trade secrets — indefinitely protected by statute "
            "(Defend Trade Secrets Act at federal level, state UTSA) and labeled as such. "
            "High-risk pattern: 'obligations survive indefinitely' without a trade secret carve-out — "
            "courts often strike the duration clause and may void the confidentiality obligation entirely."
        ),
    },
    {
        "id": "nda-remedy-01",
        "domain": "NDA",
        "topic": "remedies",
        "title": "Injunctive Relief and Remedies for Breach",
        "text": (
            "Most NDAs include language that breach will cause 'irreparable harm' justifying injunctive "
            "relief without requiring posting of a bond. While courts routinely consider this, an "
            "overly aggressive remedies clause creates risk for the receiving party: it may prevent "
            "them from raising legitimate defenses or conducting normal business during a dispute. "
            "Best practice: include injunctive relief language, but preserve the receiving party's "
            "right to contest the merits in any emergency hearing, and limit the irreparable-harm "
            "presumption to actual misuse or disclosure of confidential information — not mere "
            "technical procedural breach. Monetary damages should remain available as primary remedy "
            "for non-disclosure breaches."
        ),
    },
    {
        "id": "nda-return-01",
        "domain": "NDA",
        "topic": "return_of_information",
        "title": "Return and Destruction of Confidential Materials",
        "text": (
            "Upon request or termination, NDAs should require prompt return or certified destruction "
            "of all confidential materials within 10–30 days. Modern best practice allows retention "
            "of one secure archival copy in legal records solely for compliance, with written "
            "certification of destruction for all other copies. For digital environments: specify that "
            "electronic copies including cloud backups, email archives, and shadow IT copies must "
            "be purged or rendered irretrievably inaccessible. High risk: clauses allowing indefinite "
            "retention 'in accordance with data retention policies' without use restrictions "
            "effectively convert the NDA into a broad data access grant."
        ),
    },
    {
        "id": "nda-nonsolicitation-01",
        "domain": "NDA",
        "topic": "non_solicitation",
        "title": "Non-Solicitation Clauses Embedded in NDAs",
        "text": (
            "Some NDAs embed employee or client non-solicitation provisions. These require independent "
            "legal validity with separate consideration to be enforceable — the confidentiality "
            "obligation alone is usually insufficient consideration for a restrictive covenant. "
            "Courts scrutinize embedded non-solicitation clauses for: (1) geographic scope limited to "
            "where the parties actually do business, (2) duration typically 12–18 months maximum, "
            "and (3) breadth targeting only individuals actually contacted during the NDA period. "
            "Broad non-solicitation covering 'all employees of the company' or 'all clients' is "
            "routinely struck down. Best practice: address non-solicitation in a separate agreement "
            "with explicit, independent monetary consideration."
        ),
    },

    # ──────────────────────────────────────────────────────────────────────────
    # DOMAIN: Employment
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "emp-atwill-01",
        "domain": "Employment",
        "topic": "termination",
        "title": "At-Will Termination and the Implied Contract Trap",
        "text": (
            "At-will employment allows either party to end the relationship at any time for any lawful "
            "reason. However, implied contracts inadvertently arise from: (1) employee handbooks with "
            "progressive discipline procedures that imply termination only follows a process, "
            "(2) verbal promises during hiring ('we only fire for cause'), or (3) consistent company "
            "practices creating reasonable expectations of continued employment. "
            "High risk: employment agreements or handbooks using language such as 'permanent employee', "
            "'guaranteed position', or spelling out specific termination procedures without a clear "
            "at-will disclaimer. Best practice: explicit written at-will acknowledgment signed annually, "
            "express statement that handbooks are policies not contracts, and if 'for cause' termination "
            "is included, define 'cause' exhaustively to avoid wrongful termination claims."
        ),
    },
    {
        "id": "emp-noncompete-01",
        "domain": "Employment",
        "topic": "non_compete",
        "title": "Non-Compete Enforceability: Scope, Geography, Duration",
        "text": (
            "Non-compete clauses must satisfy a three-part reasonableness test: (1) scope limited to "
            "directly competitive activities, (2) geography limited to where the employee actually "
            "worked, and (3) duration no longer than necessary (6–18 months, rarely more than 24 months). "
            "Critical jurisdiction restrictions: California, Minnesota, Oklahoma, North Dakota ban "
            "non-competes for employees entirely. Oregon requires advance written notice plus additional "
            "consideration. FTC 2024 rulemaking targeted most non-competes nationally. "
            "Best practice alternative: use non-solicitation (limited to actual clients and employees "
            "the person worked with) combined with strong confidentiality obligations — courts uphold "
            "these far more consistently than broad geographic non-competes."
        ),
    },
    {
        "id": "emp-ip-01",
        "domain": "Employment",
        "topic": "ip_assignment",
        "title": "IP Assignment: Work-for-Hire Statutory Limitations",
        "text": (
            "Employment IP assignment clauses transferring 'all inventions' to the employer are limited "
            "by statute in multiple US states. Protected employee IP statutes (California Lab. Code §2870, "
            "Illinois 765 ILCS 1060/2, Delaware, Minnesota, NC, WA) prohibit assignment of IP developed: "
            "(1) entirely on the employee's own time, (2) without company equipment, facilities, or trade "
            "secrets, and (3) not relating to the company's current or reasonably anticipated business. "
            "High risk: 'all inventions conceived during employment' language that ignores these statutory "
            "limitations will be partially or wholly unenforceable and may expose the employer to statutory "
            "penalties. Best practice: include an invention assignment carve-out schedule for pre-existing "
            "IP; limit assignment to work-related IP created using company resources or during work hours."
        ),
    },
    {
        "id": "emp-severance-01",
        "domain": "Employment",
        "topic": "severance",
        "title": "Severance: ADEA/OWBPA Requirements and Void Releases",
        "text": (
            "Severance agreements condition payment on a general release of claims. For employees over "
            "40, ADEA/OWBPA mandates: 21-day review period, 7-day revocation window, specific ADEA "
            "waiver language, and written advice to consult an attorney. These requirements cannot be "
            "waived or shortened even by the employee. Items void in releases regardless of agreement "
            "language: (1) releases of future EEOC complaint rights, (2) releases of earned but unpaid "
            "wages or accrued PTO under most state wage laws, (3) releases of workers' compensation "
            "or disability benefits. High risk: broad 'all claims known and unknown' releases without "
            "carve-outs expose the entire release to rescission. Best practice: use EEOC-template "
            "language, tie severance to years of service formula, keep restrictive covenants separate."
        ),
    },
    {
        "id": "emp-compensation-01",
        "domain": "Employment",
        "topic": "compensation",
        "title": "Clawback and Wage Forfeiture Clauses",
        "text": (
            "Clawback clauses requiring return of bonus or commission 'upon termination for any reason' "
            "are high risk because they may constitute illegal wage deductions under state law. "
            "California, New York, and Illinois treat earned commissions as wages that cannot be "
            "clawed back after the employee has satisfied the earning conditions (e.g., closed a deal). "
            "SEC-listed companies must include Dodd-Frank Rule 10D-1 compliant clawback policies "
            "limited to executive compensation in connection with financial restatements. "
            "Best practice: limit clawbacks to (1) proven misconduct causing quantifiable harm, "
            "(2) amounts not yet vested or earned, and (3) amounts proportional to the harm. "
            "Never structure clawbacks as automatic deductions from future wages — obtain signed "
            "authorization as a separate document."
        ),
    },
    {
        "id": "emp-arbitration-01",
        "domain": "Employment",
        "topic": "dispute_resolution",
        "title": "Mandatory Employment Arbitration: Limits and Risks",
        "text": (
            "Mandatory arbitration clauses require employees to waive class action rights and resolve "
            "disputes individually. While generally enforceable post-Epic Systems v. Lewis (2018), "
            "the Ending Forced Arbitration of Sexual Assault and Sexual Harassment Act (2022) "
            "prohibits mandatory arbitration for sexual harassment and assault claims. "
            "California AB 51 (blocked federally but still litigated) targets coercive arbitration. "
            "Best practice: provide an opt-out window (30–60 days after hire), require employer "
            "to pay all arbitration costs above filing fee, specify reputable arbitration rules "
            "(AAA Employment Rules), preserve the right to seek injunctive/emergency relief "
            "in court, and exclude workers' compensation and unemployment claims. Class action "
            "waivers without these safeguards face increasing judicial and legislative challenge."
        ),
    },

    # ──────────────────────────────────────────────────────────────────────────
    # DOMAIN: Lease / Real Estate
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "lease-deposit-01",
        "domain": "Lease",
        "topic": "security_deposit",
        "title": "Security Deposit: Statutory Caps, Itemization, Return",
        "text": (
            "Security deposit rules are strictly governed by state law. Most US states cap deposits at "
            "1–3 months' rent: California (2 months unfurnished, 3 months furnished), New York (1 month), "
            "Texas (no statutory cap, reasonableness standard). Return deadlines: 14–45 days from "
            "move-out with written itemized deduction list — failure to itemize within the deadline "
            "often results in forfeiture of the entire deposit and statutory penalties (2–3x deposit). "
            "High risk for landlords: withholding for normal wear and tear (faded paint, worn carpet, "
            "minor nail holes) is routinely disallowed. Best practice: joint move-in/move-out "
            "inspection with photographic documentation, pre-agreed normal wear and tear definition "
            "in the lease, and strict adherence to the state's return timeline regardless of disputes."
        ),
    },
    {
        "id": "lease-rent-01",
        "domain": "Lease",
        "topic": "rent_escalation",
        "title": "Rent Escalation: CPI Indexing and Annual Increases",
        "text": (
            "Commercial leases commonly include annual rent increases tied to CPI or a fixed percentage. "
            "High risk for tenants: uncapped CPI indexing (CPI-U spiked 8%+ in 2022), escalation "
            "based on the landlord's sole determination of 'market rent', or cumulative step-ups "
            "that compound even when tenant is already above market rates. "
            "Best practice (tenants): negotiate a cap on annual CPI increases (typically max 3–5%), "
            "specify the exact BLS CPI-U All Urban Consumers index, include a floor at 0% "
            "(no rent reduction for deflation unless negotiated), and establish a neutral appraiser "
            "mechanism for any market-rate determination. Rent-control jurisdictions override "
            "lease terms — always verify local ordinances."
        ),
    },
    {
        "id": "lease-entry-01",
        "domain": "Lease",
        "topic": "landlord_entry",
        "title": "Landlord Entry Rights: Notice and Emergency Access",
        "text": (
            "All US states require advance notice before landlord entry (typically 24–48 hours), "
            "with immediate entry only for genuine emergencies. Residential lease clauses that waive "
            "statutory notice requirements are unenforceable in most states. "
            "High risk for tenants: clauses permitting 'reasonable' entry at any time, or for vague "
            "purposes including 'inspections or any purpose the landlord deems necessary.' "
            "Best practice: specify permitted purposes (scheduled repairs, government inspections, "
            "authorized showings), a minimum notice period (no less than state statute requires), "
            "restriction of routine entry to normal business hours (8am–6pm), and clear emergency "
            "exception requiring landlord to notify tenant as soon as practicable post-entry. "
            "Commercial tenants may negotiate tighter restrictions including security protocol "
            "compliance for sensitive facilities."
        ),
    },
    {
        "id": "lease-assignment-01",
        "domain": "Lease",
        "topic": "assignment_subletting",
        "title": "Assignment and Subletting: Recapture Traps",
        "text": (
            "Commercial lease assignment restrictions can trap tenants who need to sell their business. "
            "The standard balanced provision is 'landlord consent not to be unreasonably withheld.' "
            "However, many leases include landlord recapture rights — landlord can terminate the "
            "lease rather than approve the assignment, effectively preventing a business sale. "
            "Best practice (tenants): negotiate carve-outs for (1) corporate reorganizations, "
            "(2) mergers and acquisitions involving change of control, (3) affiliate transfers — "
            "all without requiring consent. Limit recapture rights to speculative re-letting "
            "scenarios, not business sales. Define 'reasonable' consent criteria (creditworthiness, "
            "compatible use) and negotiate profit-sharing from subletting as an alternative to "
            "outright prohibition. Absolute assignment bans materially impair business value."
        ),
    },
    {
        "id": "lease-default-01",
        "domain": "Lease",
        "topic": "default_remedies",
        "title": "Lease Default: Notice, Cure Periods, and Remedies",
        "text": (
            "Commercial lease default provisions should distinguish between monetary and non-monetary "
            "defaults with appropriate cure periods. Industry standard: monetary default (late rent) — "
            "5–10 business day cure period after written notice; non-monetary default — 30-day notice "
            "with additional 30+ days if the default cannot reasonably be cured within 30 days and "
            "the tenant is diligently pursuing cure. High risk for tenants: defaults triggered by "
            "operational changes not tied to lease breach (bankruptcy filing, change of controlling "
            "ownership), or provisions allowing termination for any technical breach without "
            "materiality threshold. Best practice: negotiate express cure rights, limit cross-default "
            "provisions to material financial obligations only, and require landlord to mitigate "
            "damages using commercially reasonable efforts rather than letting the space sit vacant."
        ),
    },
    {
        "id": "lease-restoration-01",
        "domain": "Lease",
        "topic": "tenant_improvements",
        "title": "Tenant Improvements: Restoration Obligations and TI Allowances",
        "text": (
            "Clauses requiring tenants to restore premises to 'original condition' at lease end expose "
            "tenants to restoration costs that can equal years of rent for expensive build-outs. "
            "Best practice: negotiate at lease signing — not at end of lease — which specific "
            "improvements must be removed vs. surrendered as part of the building. "
            "Landlord-funded tenant improvement allowances (TI allowances) should specify that "
            "improvements become part of the building and need not be removed. For specialty "
            "infrastructure (data centers, lab equipment): negotiate a restoration cap (e.g., "
            "'not to exceed $X per square foot') or an identified removal checklist. "
            "TI allowances structured as loans vs. grants carry different tax treatment (IRC §110) "
            "and must be documented accordingly to avoid unexpected taxable income."
        ),
    },

    # ──────────────────────────────────────────────────────────────────────────
    # DOMAIN: SaaS (Software as a Service)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "saas-sla-01",
        "domain": "SaaS",
        "topic": "uptime_sla",
        "title": "SLA Uptime: Measurable Commitments and Credit Remedies",
        "text": (
            "SaaS agreements must define uptime as a measurable percentage (not 'commercially "
            "reasonable efforts'). Industry tiers: 99.9% = 8.7 hours permitted downtime/month; "
            "99.95% = 4.4 hours; 99.99% = 4.4 minutes. Exclusions must be specific and limited: "
            "scheduled maintenance with 72-hour advance notice, customer-caused failures, and "
            "outages in third-party infrastructure outside vendor's direct control. "
            "Credit structure: 10% monthly fee for each 0.1% below SLA, capped at 30% of monthly fees. "
            "High risk: SLA with no termination right after repeated failures, token credit caps "
            "(e.g., $100 flat), or uptime measured only for a dashboard portal while API endpoints "
            "and integrations are excluded. Best practice: include 'persistent SLA failure' "
            "termination right (breach in 3+ months of any 6-month window) with full pro-rata refund."
        ),
    },
    {
        "id": "saas-data-01",
        "domain": "SaaS",
        "topic": "data_ownership",
        "title": "Customer Data: Ownership, Portability, and Deletion",
        "text": (
            "Customers must retain ownership of their data at all times — the vendor receives only "
            "a limited license to process data for service delivery. Best practice portability "
            "provisions: export within 30 days in standard machine-readable format (CSV, JSON, "
            "SQL dump), continued read-only access during a 90-day wind-down after termination, "
            "and certified deletion within 60 days of written request post-wind-down. "
            "High risk: vendor clauses permitting use of customer data for AI/ML model training "
            "without express opt-in consent, vague 'aggregate anonymized data' carve-outs that "
            "may still identify customers through inference, or deletion timelines exceeding "
            "180 days with no access during that period. GDPR note: if EU personal data is "
            "involved, a separate DPA addressing sub-processors, SCCs, and 72-hour breach "
            "notification is legally mandatory."
        ),
    },
    {
        "id": "saas-autorenewal-01",
        "domain": "SaaS",
        "topic": "auto_renewal",
        "title": "Auto-Renewal: Cancellation Windows and Price Change Traps",
        "text": (
            "B2B SaaS auto-renewal best practice: 30-day minimum cancellation notice before the "
            "renewal date (not 60–90 days, which creates unreasonable traps), price change notice "
            "at least 60 days before renewal, and a customer termination right if price increases "
            "exceed a defined threshold (e.g., 10% above the current contract rate). "
            "Consumer SaaS is additionally regulated by ROSCA and state ARL laws (California, New York, "
            "Illinois) requiring affirmative opt-in for auto-renewal, clear disclosure, and "
            "easy cancellation (as simple as sign-up). "
            "High risk: renewal for a longer term than the original (annual → multi-year auto-renewal), "
            "price increases applied at renewal without written advance notice, or cancellation "
            "requiring a phone call rather than a simple written notice or in-app action."
        ),
    },
    {
        "id": "saas-modification-01",
        "domain": "SaaS",
        "topic": "service_modifications",
        "title": "Unilateral Service Modifications and Feature Deprecation",
        "text": (
            "SaaS vendors routinely reserve the right to modify services. Industry best practice "
            "distinguishes three tiers: (1) minor enhancements — no advance notice required; "
            "(2) material feature changes — 30-day advance notice with a change log; (3) feature "
            "deprecation — 90-day advance notice with a documented migration path and customer "
            "termination right plus pro-rata refund if the deprecated feature was material to the "
            "customer's use case. High risk: 'feature parity' language allowing vendors to "
            "substitute any feature with a 'comparable' one at their sole discretion, or "
            "deprecation clauses buried in acceptable use policies updated unilaterally via "
            "website posting. Customers should require contractual service descriptions tied "
            "to specific features and APIs rather than 'current functionality' as of signing."
        ),
    },
    {
        "id": "saas-security-01",
        "domain": "SaaS",
        "topic": "data_security",
        "title": "Data Security: Standards, Breach Liability, Notification",
        "text": (
            "SaaS security provisions should reference auditable, third-party verified standards: "
            "SOC 2 Type II, ISO 27001, or sector-specific frameworks (HIPAA BAA, PCI DSS Level 1). "
            "Breach notification: 72 hours for GDPR-covered incidents; 30–72 hours for US state "
            "breach laws (most now require this). The contract must specify notification content, "
            "investigation cooperation, remediation costs, and credit monitoring obligations. "
            "High risk: data breach vendor liability capped at the same 12-month general liability "
            "cap — grossly inadequate for a large breach affecting thousands of records. "
            "Best practice: carve out data breach liability from the general cap with a separate "
            "sublimit (minimum 2–3x annual contract value), mandate vendor cyber insurance "
            "($5–10M per event minimum), and require annual penetration testing with executive "
            "summary provided to the customer upon request."
        ),
    },
    {
        "id": "saas-termination-01",
        "domain": "SaaS",
        "topic": "termination",
        "title": "SaaS Termination Rights and Refund Obligations",
        "text": (
            "SaaS termination provisions must address both for-cause and for-convenience scenarios. "
            "For-cause triggers: SLA persistent failure, material breach, insolvency, or regulatory "
            "non-compliance — 30-day notice and cure period standard, shorter for security incidents. "
            "For-convenience: customer may terminate with 30-day notice; vendor should provide "
            "pro-rata refund for prepaid but unused subscription periods. "
            "Post-termination obligations: 90-day wind-down with continued read-only access, "
            "data export assistance, and certified deletion thereafter. "
            "High risk: no customer termination right despite persistent SLA breaches, all fees "
            "non-refundable including prepaid future periods, or renewal fees due immediately upon "
            "the renewal date even if the customer submitted timely cancellation notice."
        ),
    },

    # ──────────────────────────────────────────────────────────────────────────
    # DOMAIN: Vendor / Supply Agreement
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "vendor-delivery-01",
        "domain": "Vendor",
        "topic": "delivery_acceptance",
        "title": "Delivery Terms, Inspection, and Formal Acceptance",
        "text": (
            "Vendor agreements should specify delivery terms using Incoterms 2020 (DDP, DAP, FOB) "
            "to clarify risk transfer and freight cost responsibility. Inspection periods must be "
            "appropriate to the goods: 5–10 business days for commodity goods, 30–90 days for "
            "complex technical systems requiring integration testing. "
            "High risk for buyers: automatic acceptance upon physical delivery without inspection "
            "period, or language where acceptance constitutes a complete waiver of all warranty "
            "rights including hidden defects. Best practice: separate acceptance (confirming "
            "conformity to specifications at delivery) from warranty (covering latent defects "
            "discovered post-acceptance). Partial non-conforming deliveries should trigger "
            "rejection rights for the non-conforming portion without affecting compliant items, "
            "and a written acceptance test protocol should be attached as an exhibit for "
            "complex technical procurement."
        ),
    },
    {
        "id": "vendor-warranty-01",
        "domain": "Vendor",
        "topic": "warranty",
        "title": "Product Warranties: Coverage, Duration, and Remedy Hierarchy",
        "text": (
            "Product warranties must specify: (1) coverage scope (defects in materials and workmanship "
            "under normal use and operating conditions), (2) duration (12–24 months from delivery or "
            "acceptance for most goods), (3) clear exclusions (consumables, misuse, unauthorized "
            "modification, damage from third-party products), and (4) remedy hierarchy. "
            "High risk: complete disclaimer of all warranties including implied warranties of "
            "merchantability and fitness for a particular purpose — courts may not enforce blanket "
            "disclaimers for consumer goods or when the vendor knew the buyer's specific intended use. "
            "UCC §2-316 requires disclaimers to be in writing and conspicuous (often bold or ALL CAPS). "
            "Best practice: tiered remedy structure — repair within 30 days → replace → full refund. "
            "Pass-through manufacturer warranties for third-party components should be expressly included."
        ),
    },
    {
        "id": "vendor-price-01",
        "domain": "Vendor",
        "topic": "pricing",
        "title": "Price Escalation: Index-Based Adjustments and Buyer Protections",
        "text": (
            "Long-term supply agreements must manage commodity price risk. Best practice: tie "
            "price adjustments to a specific published government index (BLS Producer Price Index "
            "for the relevant commodity category), cap annual increases at a fixed maximum "
            "(3–5% or PPI change plus 1%, whichever is lower), require 60–90 days advance "
            "written notice before any price increase takes effect, and include a buyer "
            "termination right if cumulative increases exceed a defined threshold "
            "(e.g., 10% above the original contract base price). "
            "High risk: open-ended price adjustment at vendor's sole discretion, cost pass-through "
            "clauses that allow vendors to recover all input cost increases without sharing "
            "efficiency gains or volume benefits, or price adjustment rights triggered without "
            "minimum notice. Most-Favored-Nation (MFN) pricing clauses should be negotiated "
            "when volume justifies, ensuring the buyer receives the vendor's best available price."
        ),
    },
    {
        "id": "vendor-exclusivity-01",
        "domain": "Vendor",
        "topic": "exclusivity",
        "title": "Exclusivity and Minimum Purchase Commitments",
        "text": (
            "Exclusive dealing provisions requiring buyers to source specified goods solely from "
            "one vendor create significant antitrust and commercial risk. Antitrust concern: "
            "exclusive dealing covering substantial market share in a product category can violate "
            "Sherman Act §1 and Clayton Act §3 — particularly for dominant vendors. "
            "Best practice (buyers): limit exclusivity to specific SKUs or narrow product subcategories "
            "(not entire commodity groups), cap the exclusivity period at 12–24 months with "
            "explicit renewal opt-in, link minimum purchase commitments to rolling 12-month "
            "average forecasts with force majeure relief and demand shortfall accommodation, "
            "and include immediate exit rights if the vendor fails quality, lead-time, or "
            "fill-rate standards for 2+ consecutive quarters. Never sign perpetual exclusivity "
            "clauses without robust vendor performance safeguards."
        ),
    },

    # ──────────────────────────────────────────────────────────────────────────
    # DOMAIN: General Commercial (cross-domain)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "id": "gen-cap-01",
        "domain": "General",
        "topic": "limitation_of_liability",
        "title": "Liability Cap: Structure, Adequacy, and Required Carve-outs",
        "text": (
            "Commercial agreements typically include a mutual liability cap (12 months of fees paid) "
            "and a mutual waiver of indirect/consequential/special damages. "
            "Essential carve-outs that must be excluded from the liability cap: death or personal "
            "injury caused by negligence, fraud or fraudulent misrepresentation, IP infringement "
            "indemnification, confidentiality breaches, data breaches, and gross negligence. "
            "High risk: one-sided caps protecting only the vendor; consequential damage waivers "
            "that inadvertently exclude direct losses (lost profits that are the foreseeable direct "
            "result of breach); or de minimis caps set well below actual exposure. "
            "Best practice: minimum mutual cap at 12-month fees, a separate super-cap for data "
            "breach events (2–3x annual fees), and mutual consequential damage waiver with the "
            "carve-outs listed above. Courts may void caps that shield deliberate or reckless "
            "misconduct — always include a gross negligence/willful misconduct carve-out."
        ),
    },
    {
        "id": "gen-indemnity-01",
        "domain": "General",
        "topic": "indemnification",
        "title": "Indemnification: Scope, Defense Control, and Settlement",
        "text": (
            "Indemnification clauses in commercial contracts should specify: (1) covered claims "
            "(third-party claims only unless first-party is expressly agreed), (2) trigger events "
            "(a party's own breach, negligence, IP infringement, or willful misconduct — not the "
            "other party's acts), (3) defense control (indemnifying party controls defense at their "
            "expense from day one), (4) settlement authority (indemnified party's written consent "
            "required for any settlement that includes an admission of fault or imposes ongoing "
            "obligations). High risk: unlimited indemnities for broad categories without materiality "
            "thresholds, IP indemnity without a 'knowledge' or 'reasonable belief' qualifier, or "
            "indemnities where the indemnified party must fund defense costs pending reimbursement. "
            "Best practice: mutual indemnities limited to each party's own acts; promptly tender "
            "defense costs upon notice; both parties retain approval rights over settlements."
        ),
    },
    {
        "id": "gen-termination-01",
        "domain": "General",
        "topic": "termination",
        "title": "Termination for Cause vs. Convenience: Key Protections",
        "text": (
            "Commercial contracts must clearly define: (1) termination for cause — material breach "
            "plus written notice plus 30-day cure period (shorter for payment defaults: typically 5 "
            "business days), and (2) termination for convenience — advance written notice of 30–90 "
            "days sufficient for reasonable transition. "
            "High risk: no termination for cause provision (parties must pursue breach damages only, "
            "requiring continued performance), triggers activating for minor technical breaches "
            "without a materiality threshold, or 'ipso facto' clauses triggering termination on "
            "insolvency filing — these are generally unenforceable under US bankruptcy law. "
            "Best practice: bilateral termination for cause with 'material breach' defined, "
            "graduated cure periods (immediate for safety, 5 days for payment, 30 days for "
            "operational), and defined post-termination wind-down obligations: data return, "
            "IP license termination, invoice finalization, and transition assistance."
        ),
    },
    {
        "id": "gen-amendment-01",
        "domain": "General",
        "topic": "amendment",
        "title": "Unilateral Amendment Rights and Click-Wrap Traps",
        "text": (
            "Clauses permitting one party to amend the agreement by updating a website or online portal "
            "('continued use constitutes acceptance') face increasing enforceability challenges. "
            "US courts are skeptical of unilateral amendment clauses for material terms — Uber, "
            "Airbnb, and major SaaS operators have had such clauses invalidated when customers "
            "were not adequately notified. Material terms (pricing, arbitration, liability, IP "
            "ownership) require affirmative assent that a banner notification does not provide "
            "for B2B enterprise contracts. High risk: unilateral amendments to arbitration clauses "
            "retroactively affecting pending disputes — specifically invalidated by some courts. "
            "Best practice: email notice for material changes with 30-day opt-out window allowing "
            "termination at pre-change terms, maintain a versioned change log, and require "
            "affirmative written or click-through consent for changes to fundamental terms."
        ),
    },
    {
        "id": "gen-governing-01",
        "domain": "General",
        "topic": "governing_law",
        "title": "Governing Law, Jurisdiction, and Dispute Resolution",
        "text": (
            "Choice of governing law determines which state's contract doctrines apply — implied "
            "covenant of good faith, unconscionability, and enforceability of restrictive covenants "
            "vary significantly by state. Jurisdiction clause determines where litigation must occur. "
            "Arbitration clauses waive the right to jury trial; class action waivers add further risk. "
            "High risk: mandatory arbitration in a geographically distant jurisdiction creating "
            "cost barriers that effectively block legitimate claims (unconscionability defense), "
            "governing law chosen specifically to defeat consumer or employment protection statutes. "
            "Best practice: mutually agreeable neutral jurisdiction; specify applicable arbitration "
            "rules (AAA Commercial, JAMS, or ICC for international); mandatory mediation step "
            "before arbitration (60-day period); preservation of emergency injunctive relief rights "
            "in any court; and clear allocation of arbitration costs — typically employer/vendor "
            "pays all costs above the court filing fee equivalent."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dependencies() -> None:
    """Install chromadb and sentence-transformers if missing."""
    missing = []
    try:
        import chromadb  # noqa: F401
    except ImportError:
        missing.append("chromadb")
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        missing.append("sentence-transformers")

    if missing:
        print(f"📦 Installing: {', '.join(missing)} …")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing, "-q"],
            stdout=subprocess.DEVNULL,
        )
        print("✅ Dependencies installed.\n")


def build_vector_db(reset: bool = False) -> None:
    """
    Build (or re-build) the ChromaDB collection from KNOWLEDGE_BASE.

    Args:
        reset: If True, delete existing collection and start fresh.
    """
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    print(f"📂 ChromaDB path : {CHROMA_DB_PATH}")
    print(f"🗂  Collection    : {COLLECTION_NAME}")
    print(f"📄 Total entries  : {len(KNOWLEDGE_BASE)}\n")

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # ── Handle existing collection ─────────────────────────────────────────
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        if reset:
            print("🗑  Deleting existing collection for fresh rebuild…")
            client.delete_collection(COLLECTION_NAME)
        else:
            col = client.get_collection(COLLECTION_NAME)
            count = col.count()
            print(f"ℹ️  Collection already exists with {count} entries.")
            if count == len(KNOWLEDGE_BASE):
                print("✅ Nothing to do — collection is up to date.\n")
                return
            else:
                print("♻️  Entry count mismatch — rebuilding collection…")
                client.delete_collection(COLLECTION_NAME)

    # ── Create collection with default embedding function ─────────────────
    print("🔧 Initialising embedding model (all-MiniLM-L6-v2)…")
    print("   (First run may download ~90 MB — subsequent runs are instant)\n")

    ef = DefaultEmbeddingFunction()
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Batch-insert ───────────────────────────────────────────────────────
    ids       = [rec["id"]   for rec in KNOWLEDGE_BASE]
    documents = [rec["text"] for rec in KNOWLEDGE_BASE]
    metadatas = [
        {
            "domain": rec["domain"],
            "topic":  rec["topic"],
            "title":  rec["title"],
        }
        for rec in KNOWLEDGE_BASE
    ]

    print("🚀 Embedding and inserting documents…")
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    # ── Verify ────────────────────────────────────────────────────────────
    final_count = collection.count()
    print(f"\n✅ Done! Inserted {final_count} documents into ChromaDB.\n")

    # ── Domain summary ────────────────────────────────────────────────────
    from collections import Counter
    domain_counts = Counter(rec["domain"] for rec in KNOWLEDGE_BASE)
    print("📋 Knowledge base breakdown:")
    for domain, count in sorted(domain_counts.items()):
        bar = "█" * count
        print(f"   {domain:<12} {bar} ({count})")
    print()


def test_retrieval() -> None:
    """Quick smoke-test: query each domain and print top result."""
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    ef = DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)

    test_queries = {
        "NDA":        "The receiving party shall keep all disclosed information confidential indefinitely.",
        "Employment": "Employee agrees not to compete with Employer for 3 years after termination.",
        "Lease":      "Landlord may enter premises at any time without prior notice for inspection.",
        "SaaS":       "Vendor shall use commercially reasonable efforts to maintain service availability.",
        "Vendor":     "Buyer agrees to purchase exclusively from Supplier for all product categories.",
        "General":    "Each party's liability shall be limited to fees paid in the prior 12 months.",
    }

    print("🔍 Smoke-test retrieval results:\n")
    print("-" * 60)
    for domain, query in test_queries.items():
        results = collection.query(
            query_texts=[query],
            n_results=1,
            where={"domain": domain},
            include=["documents", "metadatas", "distances"],
        )
        if results["ids"][0]:
            meta = results["metadatas"][0][0]
            dist = results["distances"][0][0]
            sim  = 1.0 - dist
            print(f"  [{domain}] → '{meta['title']}' (similarity: {sim:.2f})")
        else:
            print(f"  [{domain}] → No results found!")
    print("-" * 60)
    print("\n✅ Smoke-test complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Contract Risk Analyzer — Domain-Aware RAG Setup")
    print("=" * 65)
    print()

    _ensure_dependencies()

    reset_flag = "--reset" in sys.argv
    if reset_flag:
        print("⚠️  --reset flag detected: existing collection will be deleted.\n")

    build_vector_db(reset=reset_flag)
    test_retrieval()

    print("=" * 65)
    print("  Setup complete. Run the app with:  streamlit run app.py")
    print("=" * 65)
