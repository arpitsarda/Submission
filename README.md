# Sympera AI - Take-Home Assignment

## Files in This Submission

Before getting into the detail, here is what each file does and why I built it:

scripts/signals.py - Task 1. Computes all 8 signals from transactions.json. Nothing in it is hardcoded to Elite Builds specifically; vendor grouping and category totals are computed generically.

scripts/generate_customer_segments.py - Task 2. Sends Task 1's output to an LLM under a fixed system prompt and produces the RAG-ready customer_segments.json.

scripts/generate_hook.py - Task 3. Sends Task 2's output to identify the strongest opportunity, recommend one product, and write the RM hook.

output/signals.json - Task 1's raw output: all 8 metrics with their evidence.

output/customer_segments.json - Task 2's output: the behavioral/demographic/psychographic RAG profile we are planning to build for each company.

output/hook.json - Task 3's output: the selected signal, the product recommendation, the reasoning, and the hook.

index.html - a standalone visual mockup I built to show how an RM would actually see and use this client's profile while on a call with them. It is not part of the required deliverables, but I wanted to demonstrate what I think the end product should feel like, not just the underlying data pipeline.

requirements.txt - Python dependencies (pandas, google-genai).

A note on scalability before I get into the individual signals: I designed every signal in Task 1 to generalize beyond this one company. None of them assume a fixed vendor name, a fixed number of transactions, or a fixed category schema beyond what's documented as an assumption below. That said, I only have one month of data for one company here, so several of these signals are currently single-period observations rather than proven trends. As more months of data and more companies become available, I expect these same signal definitions to get meaningfully stronger - turning single-point comparisons into real trend lines and making it possible to build entirely new signals that one month of data simply can't support yet. I've called this out explicitly wherever it applies, rather than overstate what a single month of transactions can prove.

## Task 1 - The 8 Signals

### 1. Exposure-Adjusted Resilience Alpha

Definition: How much this company's own material spend compares to a fixed-price “shadow” baseline - what it would have paid all month if each vendor's price had stayed exactly at its Day-1 level - benchmarked against the industry's reported cost surge and weighted by how much of total spend materials represent.

Numerator: Industry benchmark (15%) minus the company's own observed material inflation rate (C_i, from the shadow-cost comparison below) - a percentage-point differential.

Denominator: None directly. The differential is scaled by a weighting factor: Material Spend ÷ Total Operating Spend.

Calculation steps: 

Group Inventory-category debit transactions by vendor, sorted by date.

For each vendor, assume quantity = 1 per transaction (no real quantity data exists) and set the baseline unit cost as the Day-1 transaction's amount (Intermountain Lumber: $12,000; Steel Supply Co: $4,000).

Shadow spend per vendor = number of transactions × baseline unit cost (Intermountain Lumber: 2 × $12,000 = $24,000; Steel Supply Co: 2 × $4,000 = $8,000; total shadow spend = $32,000).

Real spend = actual dollars paid ($23,000 + $7,800 = $30,800). Company material inflation (C_i) = (Real − Shadow) ÷ Shadow = ($30,800 − $32,000) ÷ $32,000 = -3.75%.

Differential = 15% − (-3.75%) = 18.75 percentage points. Exposure weight = Material Spend ($30,800) ÷ Total Operating Spend ($62,400) = 0.494.

Score = 18.75 × 0.494 = 9.25.

Assumptions: I assumed every material transaction represents exactly 1 unit, since no real quantity data exists - this is what turns a dollar change into a price change. I flagged this as the load-bearing assumption rather than resolve it, and I intend to replace it with real quantity data if it ever becomes available.

Thoughts on scaling: This method uses every transaction for a vendor, not just a first-vs-last comparison, and it naturally handles vendors with only one observed purchase - their shadow spend equals their real spend by construction, so they contribute zero variance without needing special-case code. The quantity = 1 assumption is the one part that needs real data to fix, not more engineering.

Why / so what: The market report frames rising material costs as the exact risk banks are watching for. If a client's own costs are moving against that trend, this is the most direct, literal rebuttal to the industry's stated concern, and the clearest basis for offering preferential terms tied to input-cost risk.

### 2. Retained Liquidity Ratio

Definition: What share of a company's observed inflows it kept as cash, in checking or moved to savings, rather than spending.

Numerator: Ending checking balance + savings transferred out.

Denominator: Total revenue inflows (all credits).

Calculation steps: 

Total credits = Revenue + Interest = $112,120.

Total expenses, including the savings transfer since it leaves checking = $67,400.

Ending checking balance = Total credits - Total expenses = $44,720.

Retained liquidity = Ending balance + Savings transfer ($5,000) = $49,720.

Ratio = $49,720 / $112,120 = 44.35%.

Assumptions: I assumed a $0 opening balance, since none was supplied - this is net cash movement within the observed period, not a verified bank balance. I also assumed the savings transfer stays within the business's control, not a personal draw.

Thoughts on scaling: This only needs credit/debit totals by category, so it works for any client as long as a savings/transfer category is identifiable.

Why / so what: A bank wants to know if a client has a real cash cushion, not just revenue. A high ratio supports both a treasury/sweep conversation and general creditworthiness.

### 3. Growth Allocation Rate

Definition: The share of total spend directed to growth-oriented categories (new equipment, technology, marketing) rather than routine operating costs.

Numerator: Growth + Tech/Growth category debits ($10,650).

Denominator: Total gross expenditure - all non-savings debits ($62,400).

Calculation steps: Sum Growth and Tech/Growth debits, divide by total non-savings debits, multiply by 100 = 17.07%.

Assumptions: I assumed transactions tagged Growth or Tech/Growth represent genuinely new capacity, not a forced replacement for broken equipment - the data alone can't distinguish these two cases.

Thoughts on scaling: Works for any category list, provided growth-type spend is tagged consistently by the source system.

Why / so what: This separates a client in survival mode from one still investing in its own capacity even under cost pressure - directly relevant to whether expansion financing makes sense to offer.

### 4. Supply Chain Concentration Ratio

Definition: The share of a company's material spend concentrated in its single largest vendor.

Numerator: Spend with the largest vendor (Intermountain Lumber, $23,000).

Denominator: Total material spend ($30,800).

Calculation steps: Group Inventory debits by vendor (parsed from the description field), sum per vendor, identify the largest, divide by the total = 74.68%.

Assumptions: I assumed vendor names are spelled consistently - no fuzzy-matching for near-duplicate names is applied yet.

Thoughts on scaling: The grouping logic has no hardcoded vendor names, so it scales to any number of vendors as-is. 

Why / so what: High concentration can mean pricing leverage through loyalty, or single-supplier dependency risk. Either way, it's a direct, named opening for a trade-finance conversation.

### 5. Labor Intensity Ratio

Definition: Share of revenue consumed by payroll.

Numerator: Payroll spend ($17,000).

Denominator: Total revenue ($112,000).

Calculation steps: Identify payroll transactions by matching "Payroll" in the description within the Operations category (no dedicated payroll category exists), sum, divide by revenue = 15.18%.

Assumptions: I assumed that all labor appears as "Payroll Service" transactions - a company using contractors tagged elsewhere would be under-counted.

Thoughts on scaling: This is the metric most exposed to inconsistent category naming. A dedicated payroll flag from the source system would remove the current text-matching dependency entirely.

Why / so what: High labor intensity signals a business that's harder to scale without linear hiring - relevant to whether payroll/banking integration would reduce the owner's admin burden.

### 6. Modernization Ratio

Definition: Technology/software spend relative to general operating spend.

Numerator: Tech/Growth category spend ($650).

Denominator: Operations spend, used as a proxy for General & Administrative spend, since no dedicated G&A category exists ($18,750).

Calculation steps: Divide numerator by denominator, multiply by 100 = 3.47%.

Assumptions: The G&A proxy is a flagged simplification.

Thoughts on scaling: Needs a real G&A category from the source system to be fully reliable across many clients.

Why / so what: Signals whether an owner is already comfortable paying for digital tools, net banking etc. - relevant to whether API banking or virtual commercial cards would land well.

### 7. Lease Inefficiency Ratio

Definition: Share of lease-plus-overhead spend going toward renting rather than owning equipment.

Numerator: Equipment-category lease payments ($2,200).

Denominator: Lease spend + Operations/G&A-proxy spend ($20,950).

Calculation steps: Divide numerator by denominator, multiply by 100 = 10.5%.

Assumptions: I assumed the Equipment category represents only leased/rented assets, distinct from the Growth category, which holds outright purchases like the equipment down payment. This only holds because this dataset happens to keep those separate.

Thoughts on scaling: At scale, this needs the source system to reliably distinguish lease payments from purchase transactions, rather than relying on this dataset's convenient category split.

Why / so what: A business bleeding cash on short-term rentals is a natural equipment-financing conversation.

### 8. Peak Cash Gap

Definition: The longest observed gap, in days, between two major client deposits.

Numerator / denominator: Not a ratio - an absolute day-count.

Calculation steps: Filter Revenue credits above a threshold ($10,000, configurable), sort by date, compute the gap between each consecutive pair, take the maximum. Result: 12 days, between March 15 and March 27.

Assumptions: I assumed deposits above the threshold represent genuine milestone payments, not refunds or interest credits.

Thoughts on scaling: The threshold should probably scale relative to a client's typical deposit size rather than stay a fixed dollar figure once this is benchmarked across companies of very different sizes.

Why / so what: Long gaps between deposits are exactly where a business is most exposed to a short-term cash crunch - a natural overdraft or invoice-factoring conversation.

Extra thought on it: We can also use the cumulative sum and make a signal for when the firm is likely to face cash crunch. 

## Task 2 - The RAG Customer Profile

I built this as a two-stage pipeline: Task 1's script produces all 8 signals as structured JSON, and I feed that directly into an LLM call governed by a fixed system prompt, rather than hand-writing the profile myself. The three sections map to specific signals as follows, so every sentence in the output can be traced back to a specific calculation above:

Behavioral (what the business does): Retained Liquidity, Supply Chain Concentration, Labor Intensity, Lease Inefficiency, Peak Cash Gap.

Demographic (what the business is): comes from a small company_profile object I supply separately - industry, location, corporate structure. These are not derived from transactions, and I was careful to keep this data source clearly separate so the model can't quietly infer facts like location from a vendor name.

Psychographic (why it behaves that way): Resilience Alpha, Growth Allocation Rate, Modernization Ratio, framed explicitly as inferences, not certainties.

Input to the model: The company_profile object, plus the full signals object from Task 1 (all 8 metrics, not a pre-filtered subset).

Output from the model: A JSON object with exactly three keys - behavioral, demographic, psychographic - each a compact array of short tagged phrases rather than flowing prose, so every one of the 8 mapped signals can be represented without blowing the token budget.

Ensuring the output stays under 300 tokens: Three things, together: I set response_mime_type to force valid JSON with no surrounding text; I gave the model 450 tokens of output headroom rather than a hard 300-token cutoff, specifically to avoid truncating valid JSON mid-object; and I instructed it directly to use short tagged phrases instead of full sentences. Critically, I don't estimate the token count - I read it directly from Gemini's own usage_metadata.candidates_token_count after generation, which is the model's actual tokenizer, not a guess. On my first real run, this returned 174 tokens, comfortably under budget. If a future run ever exceeds 300, the script prints an explicit warning rather than silently saving an over-budget file.

Exact system prompt used: 

You are a senior relationship-banking and SMB credit analyst producing a compact RAG customer profile for a bank Relationship Manager. You will receive two inputs: "company_profile" (known facts, not derived from transactions) and "signals" (8 engineered metrics from one month of transaction data). Do not invent history, trends, customer concentration, margins, liquidity, or efficiency gains not supported by the supplied data. Where a metric's own interpretation already states a limitation, preserve that caveat's substance rather than dropping it. The "behavioral" section must be a compact array with one short tagged item for EACH of these five metrics - do not omit any: retained_liquidity, supply_chain_concentration, labor_intensity, lease_inefficiency, peak_cash_gap. The "demographic" section must use company_profile only. The "psychographic" section must be a compact array with one short tagged item for EACH of these three metrics - do not omit any: resilience_alpha, growth_allocation_rate, modernization_ratio, framed explicitly as inference not certainty. Each signal appears in exactly one section, never both, and every signal must appear somewhere. Write in a formal, measured, commercially useful tone. Return valid JSON only, with exactly three top-level keys. Keep the entire JSON response under 300 tokens total. No filler, no repeated phrasing across sections.

## Task 3 - Prompt Engineering and the RM Hook

Task 3's job is to turn the segmentation profile into one specific product recommendation and a hook the RM can actually use. 

Input to the model: Only customer_segments.json - Task 2's output. Nothing from Task 1's raw signals goes in; the RAG file is meant to be a self-sufficient artifact I can point to at scale, so Task 3 has to work from it alone. Everything the model needs to interpret that file - what each field means, and how to judge which item is the strongest signal - now lives inside the system prompt itself.

Output from the model: A JSON object with four keys - primary_signal (which signal it selected), product_recommendation (one named product), reasoning (2-3 sentences connecting the signal to the product), and hook (the 2-sentence line itself).

Why magnitude is a guide, not a script: I moved the ranking logic out of my own code and into the system prompt's instructions, so the model compares the percentage or ratio values stated in each tagged item directly, instead of relying on a pre-computed list I hand it. It's still told it can override the largest-magnitude item if another is more commercially actionable - the goal was to keep the selection genuinely data-driven without pre-deciding the answer in Python.

Exact system prompt used: 

You are a senior relationship-banking strategist. You will receive ONLY a customer's segmentation profile (customer_segments.json) - three keys: "behavioral", "demographic", "psychographic". There is no separate raw-data file; treat this JSON as your complete evidence base.

Field meanings, so you can interpret the tagged items correctly: Behavioral items describe observed spending/cash-flow patterns - retained liquidity (cash kept vs. spent), supply chain concentration (spend share with the largest vendor), labor intensity (payroll as % of revenue), lease inefficiency (leasing vs. owning equipment), peak cash gap (longest gap in days between major deposits). Demographic items are firmographic facts only (industry, location, structure) - never a source of financial signal. Psychographic items are inferred traits, not certainties - cost resilience (material costs vs. industry benchmark), growth allocation (spend directed to growth/tech), modernization (tech spend vs. overhead).

How to judge magnitude: each behavioral/psychographic item states a percentage, ratio, or day-count. Compare these values directly - the item with the largest absolute percentage or ratio is generally the strongest signal (day-counts like peak cash gap are a different unit and should only be chosen over a percentage-based item if the business story is clearly more urgent). You may override the largest-magnitude item if another is more commercially actionable given the full picture - magnitude is a starting signal, not a rule.

Task: 1) Identify the single item that represents the most significant, actionable gap or opportunity for this client, using the magnitude guidance above. 2) Recommend exactly ONE specific, named bank product that follows logically from that item. Do not pick a product first and justify it after - the evidence must come first in your reasoning. 3) Write a 2-sentence hook a Relationship Manager could say on a call or type at the top of an email. Sentence 1 states the observed evidence. Sentence 2 makes the specific ask or offer. Ground every number in the hook in the actual supplied data - never invent a figure. Do not claim trends, sustained performance, or anything not supported by one month of observed data. Return valid JSON only, with exactly these top-level keys: primary_signal, product_recommendation, reasoning, hook. No markdown, no extra keys.

## Products Considered

Across all 8 signals, here is the full set of bank products I mapped to a specific signal - not a generic product list, but one product per identified gap:

Supplier Payables Financing / Trade Finance - tied to Supply Chain Concentration.

Treasury Sweep & Reserve Management - tied to Retained Liquidity.

Integrated Payroll & Banking - tied to Labor Intensity.

Preferred Working-Capital Line - tied to Resilience Alpha.

Equipment Ownership Term Loan - tied to Lease Inefficiency.

Overdraft Facility / Invoice Factoring - tied to Peak Cash Gap.

Business Expansion Term Loan - tied to Growth Allocation Rate.

## Signals I Haven't Built Yet

A few signal ideas came up while I was working through this that I haven’t built yet. I'm listing them here because I think knowing what to build next is as important as what I did build:

Fire insurance for inventory - the company holds physical material inventory (lumber, steel); once inventory levels are visible the way they are here, insurance protecting against loss of that inventory becomes a natural cross-sell.

Workman's compensation insurance - payroll is present and material ($17,000/month), which implies salaried or hourly labor that would need this coverage. 

Client bidding assistance / bank guarantees - construction firms bidding on new projects often need a bank-issued guarantee or surety instrument. This becomes visible once bidding activity is observable, which this one month of settled transactions doesn't show.

FX services - if a company's transactions ever showed a foreign-currency vendor or an international project payment, that would be a clear signal to build against. 

## The Sample Dashboard View

Separately from the required deliverables, I built index.html - a sample view of how an RM would actually see this client while on a call with them. It includes a client list sidebar, a top bar to search clients and select a reporting period, a contact card for the client's primary contact, the hook line phrased so it can be read aloud directly rather than sounding like a data summary, the supporting metrics behind that hook, the other pitches ranked by signal strength, the full transaction ledger for evidence, and a collapsible "ask about this client" panel. I built this because our own product is fundamentally this kind of RM-facing interface, and I wanted to show what I think the finished experience should feel like, not just the data pipeline behind it.
