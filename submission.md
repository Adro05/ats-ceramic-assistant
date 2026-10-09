# Task 3 — Submission Form
## Client A — Ceramic Tiles

### 1. In two sentences: what is the client's real problem, and how is it different from what they asked for?

The client wants to eliminate a 6–7 hour batch-change shade-correction loop in which QC manually changes the print file, fires multiple variants, measures them, and repeats while production continues. The real problem is not simply comparing images or automating Photoshop: it is separating an immutable canonical master from batch-specific production variation and using measured colour/gloss evidence to determine whether a bounded printer-side correction is justified.

### 2. Who are the people in this deal, and what does each of them need to hear from you?

- **Head of IT/OT Procurement:** Needs a clear commercial case, delivery scope, pricing assumptions, dependencies, and a credible pilot path without committing to unverified plant-specific integration.
- **Special Projects Lead — Procurement:** Needs confidence that the proposal is pragmatic, can reduce the six-to-seven-hour correction loop, and will produce measurable operational value without becoming a science project the plant rejects.
- **Plant QC Manager:** Needs objective measurement-based recommendations, explicit confidence and escalation behaviour, and assurance that human QC remains the approval authority.
- **Plant Design / Pre-Press Team:** Needs the canonical master protected from batch-specific compensation and a workflow that reduces repeated Photoshop trial-and-error rather than introducing another opaque editing tool.
- **Production Line Superintendent:** Needs the correction process to fit batch-change operations without unnecessarily disrupting throughput, with process-side or unsupported cases stopped rather than forced into a printer correction.

### 3. The five questions you would ask on the next call, ranked. For each one: why it matters, and what you assumed while you wait for the answer.

**1. Which reference/master file is authoritative for each SKU, and how are historical working references related to it?**  
**Why it matters:** If the current working file has accumulated batch-specific edits, using it as ground truth would reproduce the drift we are trying to remove.  
**Working assumption:** We can establish or designate an immutable canonical master for the pilot SKU set and retain historical files only as lineage/evidence.

**2. Which printer controls are actually available and safe to change for a batch correction, and how are those controls exposed through the current workflow/RIP?**  
**Why it matters:** The optimiser must only use confirmed controllable inputs; otherwise the system could recommend changes the plant cannot actually apply.  
**Working assumption:** A bounded set of printer-side inputs can be identified during discovery, but no specific printer, RIP, channel count, file format, or API is assumed until confirmed.

**3. What colour and gloss acceptance criteria does Plant QC use today, and which measurement geometry/instrument is authoritative?**  
**Why it matters:** We need a defensible definition of a successful correction and must separate colour mismatch from gloss/process-side effects.  
**Working assumption:** The existing lab can provide repeatable L*, a*, b* measurements and gloss readings, and Plant QC can confirm the production acceptance criterion before controlled use.

**4. How repeatable are the current lab measurements, including across repeated measurements and relevant tile regions?**  
**Why it matters:** A noisy measurement can make a correctable batch look like a printer problem and cause the system to recommend a false correction.  
**Working assumption:** A short replicate/gauge study will establish the measurement noise floor before calibration data are used for modelling.

**5. What evidence distinguishes printer-correctable variation from process-side variation such as kiln, glaze, or material effects?**  
**Why it matters:** If the observed deviation cannot be explained by the validated printer controls, the safe action is escalation rather than blind correction.  
**Working assumption:** The pilot can use colour/gloss evidence, calibration coverage, and held-out validation to classify cases as correctable, near-boundary/OOD, process-side, or indeterminate.

### 4. Which options did you consider, and why did you pick the one you did? What did you rule out, and why?

I considered four approaches:

1. **Camera/video analytics:** useful for monitoring appearance, but not the right Phase 1 control mechanism for calibrated ceramic colour. It also does not solve the Golden Master drift created by editing artwork.
2. **First-principles kiln/raw-material modelling:** potentially valuable for root-cause analysis, but it requires plant/process variables that are not yet available and creates a much larger OT/data-science project.
3. **Manual-only process improvement:** could standardise the existing workflow but would not remove the repeated trial-and-error calculation that creates the six-to-seven-hour loop.
4. **Batch-specific, measurement-driven correction using the existing lab workflow:** recommended. It directly addresses the client's operational bottleneck while preserving the canonical master and keeping the solution bounded by measured evidence and confirmed controls.

I deliberately would **not** build continuous camera monitoring, autonomous production release, automatic modification of the canonical master, automatic retraining, or kiln/raw-material causal modelling in Phase 1. I also would not commit to a particular RIP integration or invent printer channels/tolerances before discovery.

### 5. How will the client know the pilot worked? Metric, baseline, threshold, how many runs, who measures, and what result would make you stop.

**Baseline:** The current correction process takes approximately 6–7 hours and relies on repeated physical variants and manual trial-and-error.

**Proposed FDE pilot gates — planning assumptions subject to Plant QC confirmation:**

- **Operational efficiency:** At least 90% of accepted measurements should produce a QC-ready recommendation within 10 minutes of accepted measurement; the target is no more than one verification firing per correction case.
- **Colour quality:** Measure per-region ΔE00 and gloss after verification. At least 90% of controlled corrections should meet the client-approved colour acceptance criterion; the numerical production tolerance is confirmed with Plant QC before controlled use.
- **Model validity:** Use approximately 10–12 controlled batch changes across multiple production runs, with a held-out validation subset. Increase the sample if calibration coverage is insufficient.
- **Safety/trust:** 100% of production recommendations require human QC approval. Unsupported/OOD and process-side cases must be blocked or escalated rather than forced into correction.

**Who measures:** Plant QC owns the production measurement and sign-off; the FDE team analyses the measurements and records the evidence.

**Stop/hold criteria:** Stop or escalate if measurement repeatability is inadequate, colour quality deteriorates versus the existing process, unsupported/OOD cases are incorrectly recommended, process-side evidence dominates, verification repeatedly fails, or the minutes-scale decision target is not achieved after calibration.

### 6. What does this cost the client, and what does it cost us to deliver? Show the arithmetic and name the pricing structure. What value does the client get, in their money?

**Illustrative delivery-cost assumption:**

- FDE: ₹1.0 lakh/week × 11 active delivery weeks = **₹11.0 lakh**
- Colour-science/calibration engineer: ₹1.2 lakh/week × 2.5 weeks = **₹3.0 lakh**
- Integration engineering: ₹1.0 lakh/week × 0.75 weeks = **₹0.75 lakh**
- Estimated direct delivery cost = **₹14.75 lakh**

These are planning assumptions, not client-provided rates.

**Proposed commercial structure:** Fixed-fee single-line discovery/calibration pilot at **₹20 lakh**. **Payment structure (proposed):** 30% at kickoff, 40% after the validated calibration/shadow milestone, and 30% on delivery of the final pilot report. The difference between the illustrative direct delivery cost and the proposed price covers project management, QA, contingency, and commercial risk.

**Indicative running-cost assumption:** approximately **15% of the pilot price = ₹3 lakh/year**, plus any plant-specific infrastructure or integration maintenance. Actual running cost depends on the plant's security, deployment, support, and integration requirements.

**Client value model:**  
Annual value = `52 × changeovers/week × [(off-shade m²/changeover × realised loss/m²) + (avoided correction hours/changeover × value/hour) + (avoided QC/pre-press hours/changeover × loaded labour rate/hour)]`.

I would quantify those inputs during discovery rather than invent production volumes or loss rates. The commercial scale gate is that measured annualised value should be at least **3× the expected annual support/run-rate cost**, alongside the operational, quality, safety, and trust gates above.

### 7. Timeline. What is the first result the client sees, and when? Which steps depend on the client rather than on us?

**Weeks 1–2 — Discovery:** Confirm master lineage, printer/RIP controls, instrument setup, measurement repeatability, current workflow, acceptance criteria, and plant dependencies.

**First client-visible result — by the end of Week 2:** The client receives a measurement-quality and diagnosability report showing whether the master/batch measurements are sufficiently repeatable and whether observed deviations appear printer-correctable, process-side, or indeterminate.

**First engineering recommendation — by Weeks 3–5:** After the initial validated calibration experiment, the client sees a bounded correction recommendation in shadow mode.

**Weeks 3–5 — Calibration:** Run controlled calibration experiments and produce the first validated/shadow-mode correction recommendation.

**Weeks 6–8 — Shadow pilot:** Compare recommendations with current QC decisions without autonomous production action.

**Weeks 9–11 — Controlled pilot:** Evaluate agreed batch changes with human QC approval and verification.

**Week 12 — Decision:** Review pilot gates, economics, risks, and whether production rollout is justified.

**Client-dependent steps:** access to the relevant plant/QC team; authoritative master/reference files; measurement and instrument access; confirmation of controllable printer inputs; acceptance criteria; calibration/test windows; QC participation; and RIP/interface documentation if integration is pursued.

### 8. Where did you push back on the client, narrow their ask, or tell them not to do something?

I pushed back on treating this as a computer-vision problem and on automatically editing the master artwork. The canonical master should remain immutable; batch-specific compensation should be a separate, versioned correction layer.

I also narrowed the Phase 1 scope by excluding continuous camera monitoring, autonomous production release, automatic retraining, and kiln/raw-material causal modelling. I would not promise a specific RIP integration, printer channel set, or colour tolerance until those facts are confirmed. Finally, I would interpret the client's “minutes, not hours” requirement as a target for the analytical/recommendation step rather than pretending the physical firing cycle itself can be eliminated.

### 9. What is most likely to go wrong with your plan, and what in these notes did you not trust?

The most likely failure is that measurement noise or process-side variation makes a printer correction appear more reliable than it really is. Gloss, surface texture, kiln/material variation, insufficient calibration coverage, or an unseen operating condition could make the response model invalid outside its supported envelope.

I did not trust the assumption that getting the first sample of a batch right guarantees that the entire batch will remain correct; that needs verification rather than being treated as a fact. I also would not assume that the current working reference is the authoritative Golden Master until file lineage is confirmed. The system therefore fails closed: unreliable measurements are rejected, process-side/indeterminate cases are escalated, OOD cases are blocked, QC approval is required, and verification failures trigger bounded iteration or escalation.

### 10. What did you use AI for? Which tools, where they helped, where they misled you, and what you threw away?

I used **ChatGPT and Claude** for discovery-note decomposition, stakeholder and option structuring, technical reasoning, critique of the case study, and drafting/revising written deliverables. I used **Python** for synthetic-data generation, prototype/document generation, testing support, and validation of the submission artefacts.

AI helped accelerate the transition from messy discovery notes to a structured FDE plan and helped identify gaps in the evaluation, commercial, and stakeholder sections. It initially pushed me toward a more architecture-heavy solution than the client problem required and suggested overly specific assumptions such as camera/vision approaches, premature deep-learning use, and unverified printer/RIP details.

I threw away those unsupported approaches, along with invented production tolerances, unsupported economic claims, autonomous release assumptions, and any design that would modify the canonical master. The final approach keeps the empirical response model as the primary decision mechanism, uses ML only as a separately gated residual evaluation, and keeps human QC and verification in the loop.

### Optional GitHub Repository

https://github.com/Adro05/ats-ceramic-assistant
