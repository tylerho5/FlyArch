## Region profiles and preregistered XGBoost expectations

Neuropil membership follows `src/regions.py`. Feature-group order is: type-pair, NT, innervation cosine, distance, same-side, degree.

| Region | Neuropils | Main function and motif | Expected feature ranking |
|---|---|---|---|
| AL | AL | Olfactory glomerular processing: ORNs, PNs, LNs, and local inhibitory/excitatory circuits. (eckstein2024_neurotransmitter) | type-pair > innervation > NT > distance > degree > same-side |
| CX | FB, EB, PB, NO | Columnar and ring-attractor circuits for navigation and state variables; tangential, columnar, and output neurons. (eckstein2024_neurotransmitter) | innervation > type-pair > distance > NT > degree > same-side |
| GNG | GNG | Mixed sensory, ascending, descending, and motor gateway circuitry [general]. | type-pair > degree > NT > distance > innervation > same-side |
| INP | CRE, SCL, ICL, IB, ATL | Mixed inferior-protocerebral sensory and integrative pathways [general]. | type-pair > innervation > distance > NT > degree > same-side |
| LH | LH | Higher-order olfactory and multimodal integration, including PN-to-output motifs [general]; pheromone-related outputs are concentrated here. (berg2025_male_cns_sexual_dimorphism) | type-pair > NT > innervation > distance > degree > same-side |
| LX | BU, LAL, GA | Action-selection and premotor integration around lateral accessory and gall-associated circuits [general]. | innervation > type-pair > distance > NT > degree > same-side |
| MB | MB_CA, MB_PED, MB_VL, MB_ML | Learning and memory: Kenyon-cell intrinsic circuitry with MBON and DAN compartments. | type-pair > NT > innervation > degree > distance > same-side |
| OL | LA, ME, AME, LO, LOP | Retinotopic visual columns: lamina, medulla, lobula, and lobula plate pathways. (matsliah2024_visual_system_partslist) | distance > innervation > type-pair > same-side > NT > degree |
| PENP | SAD, AMMC, FLA, CAN, PRW | Periesophageal and mechanosensory gateway circuits [general]. | type-pair > distance > innervation > NT > degree > same-side |
| SNP | SLP, SIP, SMP | Superior protocerebral multimodal association and neuromodulatory circuits [general]. | type-pair > NT > innervation > distance > degree > same-side |
| VLNP | AOTU, AVLP, PVLP, PLP, WED | Ventrolateral visual, mechanosensory, and multimodal integration [general]. | innervation > type-pair > distance > NT > degree > same-side |
| VMNP | VES, EPA, GOR, SPS, IPS | Ventromedial sensory, visceral, and integrative pathways [general]. | type-pair > NT > distance > innervation > degree > same-side |

## F1 data facts

Density is directed edges divided by `N(N−1)`. The processed F1 files contain `graph.json`, `schema.json`, and selected `extras.npz` files. The schema exposes a shared 8,840-entry cell-type vocabulary and six NT classes, but does not contain region-level realized type counts, NT frequencies, or side assignments. Therefore those quantities are reported as unavailable rather than inferred.

| Region | Neurons | Edges | Density | Synapses | Cell types | NT mix | Bilateral fraction |
|---|---:|---:|---:|---:|---:|---|---|
| OL | 93,270 | 1,466,840 | 0.0169% | 27,761,365 | 8,840-vocabulary; realized count unavailable | 6-class vocabulary; mix unavailable | unavailable |
| VLNP | 20,179 | 361,760 | 0.0888% | 7,371,054 | same | unavailable | unavailable |
| SNP | 14,428 | 157,849 | 0.0758% | 3,409,222 | same | unavailable | unavailable |
| VMNP | 10,737 | 142,711 | 0.124% | 3,087,077 | same | unavailable | unavailable |
| GNG | 8,873 | 118,094 | 0.150% | 2,713,213 | same | unavailable | unavailable |
| INP | 13,480 | 105,222 | 0.0579% | 2,251,293 | same | unavailable | unavailable |
| CX | 3,219 | 70,830 | 0.684% | 1,347,690 | same | unavailable | unavailable |
| PENP | 7,337 | 68,078 | 0.126% | 1,337,183 | same | unavailable | unavailable |
| AL | 3,827 | 67,576 | 0.462% | 1,487,993 | same | unavailable | unavailable |
| MB | 8,660 | 65,557 | 0.0874% | 1,535,281 | same | unavailable | unavailable |
| LH | 5,783 | 59,218 | 0.177% | 1,267,126 | same | unavailable | unavailable |
| LX | 3,694 | 40,508 | 0.297% | 910,210 | same | unavailable | unavailable |

Sources: each region’s processed `graph.json`/`schema.json`.

## Preregistered expectations

These are predictions made before SHAP results exist.

- **OL:** geometry and innervation should dominate because retinotopic columns impose strong spatial and projection-pattern constraints.
- **CX:** innervation should dominate because layer/column targeting is structured beyond simple Euclidean distance.
- **AL:** type-pair encoding should dominate because glomerular neuron classes impose stereotyped connection probabilities.
- **MB:** type-pair and NT should dominate because Kenyon-cell, MBON, and DAN compartments have distinct wiring and transmitter roles.
- **LH:** type-pair and NT should dominate because output partners are functionally specialized.
- **LX:** innervation should outrank NT because LAL/BU/GA circuits are shaped by projection patterns.
- **GNG:** type-pair and degree should dominate because the region mixes many peripheral and motor classes with broad degree variation.
- **INP, SNP, VMNP, PENP, VLNP:** type-pair should generally dominate, with innervation or geometry second, because these are heterogeneous association and gateway regions lacking one single repeated microcircuit.
- **Same-side** is expected to be weakest overall except in paired regions with strong ipsilateral wiring.
- **Degree** is expected to matter most in heterogeneous gateway regions, but it is a predictive control variable rather than a circuit-specific mechanism.

## Main cross-region contrast

The strongest predicted contrast is **OL versus CX**:

- OL should be most geometry/innervation-led because of retinotopic column structure.
- CX should be more innervation/type-led because columnar and layer-specific circuits are not reducible to physical distance.

This contrast would be falsified if OL’s SHAP ranking is led by type-pair or degree while CX is led by distance, with the difference replicated across seeds and evaluation suites. tester