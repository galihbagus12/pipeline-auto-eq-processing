# From Phase Picks to a Verified Earthquake Catalogue

**Pipeline stage 2 — `association.py`**

This document explains how loose phase picks are turned into confirmed earthquakes,
and why each filtering step exists. It covers the association process, the five
quality-control tests, and the outputs produced for hypocentre location.

---

## 1. The problem this stage solves

The picking stage (`pick.py`) scans every station independently and marks anything
that looks like a P or S wave arrival. It produces **132,634 picks** across 156 days
and 23 stations. It has no concept of an earthquake — only of arrivals.

Association is the step that groups those isolated arrivals into events. This is
where the central ambiguity lives:

> When several stations record arrivals within a short time window, are we looking at
> **one earthquake recorded by many stations**, or at **several unrelated events that
> happened to occur close together in time**?

Getting this wrong is costly in both directions. Merging two distinct earthquakes
produces a catalogue entry that corresponds to no real event, and its location will be
somewhere between the two. Splitting one earthquake discards stations that would have
constrained its position.

The pipeline therefore treats association as a *hypothesis* and subjects every
candidate event to physical tests that a genuine earthquake must pass.

---

## 2. Association: building candidate events

**Tool:** PyOcto (octree search over 4D space — latitude, longitude, depth, origin time)

The associator proposes a trial hypocentre and origin time, predicts when each wave
should arrive at each station using an assumed velocity model, and counts how many
observed picks match those predictions. Trial points that explain many picks
simultaneously become candidate events.

### The assumed velocity model

| Parameter | Value |
| --- | --- |
| P-wave velocity (Vp) | 6.0 km/s |
| S-wave velocity (Vs) | Vp / 1.73 = 3.468 km/s |

A single homogeneous velocity is a deliberate simplification. The Earth is layered —
the crust is slower (~5.8 km/s) and the mantle faster (~8.04 km/s) — so no single value
fits both near and distant stations exactly. That mismatch is absorbed by the
**tolerance** parameter described below.

### Every search parameter is derived from the station layout

Nothing here is hand-tuned, so the pipeline adapts itself to any network. All values
follow from one measured quantity: the **aperture**, the largest distance between any
two stations.

| Derived quantity | Rule | This network |
| --- | --- | --- |
| Aperture | largest inter-station distance | **284.35 km** |
| Association cutoff | aperture × 1.2 | 342 km |
| Maximum search depth | 0.7 × aperture, clipped | 95 km |
| Search area padding | 0.15 × aperture | 42.7 km |
| Origin search window | R_max / Vs × 1.5 | 160 s |
| **Velocity tolerance** | 0.10 × (R_max / Vp) | **5.92 s** |

### Why the tolerance matters more than anything else

The tolerance is the time budget allowed between an observed arrival and the predicted
one. Picks outside the budget are ignored, which can push an event below the minimum
station requirement and discard it entirely — even when the earthquake is real.

The relationship between tolerance and yield is **humped**, not monotonic. Measured on
a 5-day test subset:

| Tolerance | Events found | Runtime |
| --- | --- | --- |
| 2.0 s | 1 | 0.7 s |
| 3.0 s | 2 | 2.5 s |
| 4.0 s | 2 | 7.1 s |
| 5.0 s | 2 | 16.4 s |
| **6.0 s** | **2** | 35.4 s |
| 8.0 s | 1 | 114.5 s |

Too tight and distant stations are dropped, so real events fail the minimum-station
test. Too loose and unrelated noise picks appear compatible with any trial location, so
spurious high-count solutions outcompete the correct ones — and runtime explodes. The
automatic rule sits deliberately near the top of the safe range, because the philosophy
of this pipeline is to let later stages do the filtering rather than lose real events
early.

### Minimum requirements for a candidate event

- At least **6 picks** in total
- At least **3 different stations** with a complete P *and* S pair

Association runs **one calendar day at a time**. Events last seconds to minutes, so a
day boundary never splits one, and the date becomes a natural resume point.

---

## 3. Quality control: five independent tests

Every candidate event is then tested. The tests are independent by design — each one
probes a different physical property, so an event that satisfies all five is unlikely
to be a coincidence.

### QC-0 — Structural integrity

Four cleanup rules applied before any physics:

| Rule | Action |
| --- | --- |
| **QC-0a** | Remove duplicate picks (same station, same phase, within 2 s) — keep the most confident |
| **QC-0d** | Remove stations contributing only one phase (P without S, or S without P) |
| **QC-0b** | Reject events lacking either P or S entirely |
| **QC-0c** | Re-check that at least 3 distinct stations still have a complete P-S pair |

Single-phase stations are removed rather than repaired. Estimating a missing S arrival
from the hypocentral distance would derive that value from the very location being
solved for; feeding it forward would let the location appear to confirm itself.

QC-0c is re-applied at the end because the earlier removals can reduce the pair count
below the minimum the associator originally guaranteed.

### QC-1 — Is the event within reach of the network?

For a given station, the delay between the S and P arrivals grows with distance:

```
distance = K × (Ts − Tp)          where  K = Vp·Vs / (Vp − Vs) = 8.2192 km/s
```

An event too far away to be sensibly located by this network produces an implausibly
large S−P delay. The bound is derived, not chosen:

```
max(Ts − Tp) = aperture / K = 284.35 / 8.2192 = 34.60 s
```

Any event containing a station pair beyond this — or with S arriving before P — is
rejected.

### QC-2 — Is the geometry physically possible?

For each pair of stations that recorded the same event, the distance each one implies
must be consistent with how far apart those two stations actually are. This is the
**triangle inequality**:

```
| d_A − d_B |  ≤  D_AB
```

where `d_A`, `d_B` are the distances implied by each station's S−P delay, and `D_AB` is
their true physical separation. If station A says the event is 20 km away and station B
says 150 km, but the two stations sit only 37 km apart, no single point in space can
satisfy both. They cannot have recorded the same earthquake.

**The distances used here come from the picks alone**, not from the computed location.
This is essential: distances derived from a single hypocentre automatically satisfy the
triangle inequality, so the test would be vacuous.

**Threshold.** Rather than cutting at a percentile of the observed distribution, the
threshold is physical, with an allowance for measurement error:

```
reject when   |d_A − d_B| − D_AB   >   n·σ
σ = K × 2 × σ_pick = 8.2192 × 2 × 0.3 s ≈ 4.93 km
threshold (n = 3) ≈ 14.79 km
```

A percentile threshold would behave badly here for two reasons. It always removes a
fixed fraction of events regardless of data quality — cutting at P95 discards 5% even
if every event is perfect. And it penalises the best-recorded events: an event seen by
8 stations has 28 station pairs and is therefore far more likely to have one pair
exceed the percentile than an event seen by 3 stations with only 3 pairs. The physical
threshold has none of these properties: it rejects nothing when the data are sound.

On the 30-day validation run this is exactly what happened — the median excess was
**−52.9 km**, far inside the geometric bound; only 2 of 97 station pairs (2.1%) exceeded
the bound at all, and **none** exceeded the error tolerance.

*(Note: a median is a measure of central tendency, not a rejection threshold — cutting
at P50 would discard half the data. The median does have a proper role in this
pipeline, as the robust centre in QC-4 below.)*

### QC-3 — Are the fitted velocities physically possible?

Two straight-line fits are made per event:

| Fit | Relationship | Physical meaning of the slope |
| --- | --- | --- |
| Travel-time | `Tp` vs hypocentral distance | slope = 1 / Vp |
| Wadati | `(Ts − Tp)` vs `Tp` | slope = Vp/Vs − 1 |

A **negative or zero slope** is rejected. On the travel-time fit it would mean arrivals
getting *earlier* with distance; on the Wadati fit it would mean S waves travelling
faster than P waves. Neither is possible.

Only the slope is tested. The intercept is deliberately not used: `Tp` is measured
relative to the associator's origin time, so if that origin is late by δ seconds the
Wadati intercept shifts by roughly `−(Vp/Vs − 1)·δ`. A slightly negative intercept
therefore measures the associator's origin-time precision, not the quality of the picks.
Applying a hard zero-intercept cut discarded 21 of 24 events in testing, including a
14-station event with an excellent fit (Vp = 6.09 km/s, Vp/Vs = 1.759, R² = 0.997)
rejected over an intercept of −0.57 s — equivalent to an origin time 0.8 s late.

### QC-4 — Do all stations agree on when the earthquake happened?

This is the test that directly answers the ambiguity from Section 1, and the only one
that can catch two distinct events merged into one.

Each station with a P-S pair can compute the origin time **by itself**:

```
OT_i = Tp_i − (Ts_i − Tp_i) / (Vp/Vs − 1)
```

The S−P delay tells that station how far away the source is; knowing the distance, it
can work backwards from its own P arrival to when the rupture began. If every station
recorded the same earthquake, all these independent estimates must agree.

**Worked example.** Four stations recording one event at t = 0, plus one station that
actually recorded a different earthquake 20 seconds later:

| Station | Own origin-time estimate | Deviation |
| --- | --- | --- |
| EJA12 | 0.00 s | 0.00 |
| EJA18 | 0.00 s | 0.00 |
| EJA24 | 0.00 s | 0.00 |
| EJA13 | 0.00 s | 0.00 |
| **EJA15** | **20.00 s** | **20.00** ← |

The deviation equals the actual time separation of the two earthquakes. QC-2 does not
catch this case: two different events can easily produce a distance set that satisfies
the triangle inequality perfectly.

**Two properties make this test strong.** First, absolute Vp cancels out of the formula
— only the Vp/Vs *ratio* remains — so the test is insensitive to the crude homogeneous
velocity assumption. The ratio is far better constrained (typically 1.70–1.78) than
absolute velocity. Second, the centre is estimated with the **median** and the spread
with the **MAD** (median absolute deviation), both robust to outliers, so a single
deviating station cannot drag the reference toward itself.

```
threshold = max( 2.0 s ,  3 × 1.4826 × MAD )
```

The 2.0 s floor comes from error propagation: a pick uncertainty of 0.3 s becomes about
0.82 s in the origin-time estimate — amplified roughly threefold by the `1/(Vp/Vs − 1)`
term — so the floor sits at about 2.4σ. It also protects events with few stations,
where the MAD itself is poorly determined.

---

## 4. Validation: the catalogue checks itself

Two pooled diagrams provide an independent verification that does not depend on any
individual QC test. Every accepted station observation from every event is plotted
together, and a single global line fitted.

If the associations were noise, these points would scatter. Instead:

| Diagram | Fitted from data | Assumed in the model |
| --- | --- | --- |
| Pseudo-distance | **Vp = 6.096 km/s** (R² = 0.9951) | 6.0 km/s |
| Wadati | **Vp/Vs = 1.755** (R² = 0.9683) | 1.73 |

The velocity structure recovered from the catalogue matches the values assumed at the
start — even though those values were never imposed on the fit. This is the strongest
single indication that the events are real earthquakes.

Both diagrams are produced twice: once over **all associated events** and once over
**events passing QC**, so the effect of quality control is directly visible.

---

## 5. Results

Validation run over 30 days (2015-12-01 to 2015-12-30, 41,742 picks):

```
Candidate events from association     24
  rejected by QC-1 (S−P too large)     2
  rejected by QC-4 (origin-time)       5
  rejected by QC-2 (geometry)          0
  rejected by QC-3 (velocity sign)     0
                                    ────
Final catalogue                       17 events, 110 phase picks
```

Characteristics of the accepted events:

| Property | Range |
| --- | --- |
| Stations per event | 3 – 7 |
| Complete P-S pairs | 3 – 7 |
| Depth | 0.7 – 94.3 km |
| Median azimuthal gap | 280.6° |

The large azimuthal gap is expected at this stage. Most events are recorded by only 3–4
stations, all on one side of the epicentre, so depth in particular is poorly resolved —
several solutions sit exactly on the search-grid boundary. **This is not a defect of
the catalogue but a limit of the associator**, which performs a coarse grid search with
a single homogeneous velocity. Precise hypocentres come from the next stage, NonLinLoc,
using a layered velocity model and a full probabilistic search.

---

## 6. Outputs

### The catalogue

| File | Contents |
| --- | --- |
| `csv/event_summary.csv` | One row per event: origin time, latitude, longitude, depth, contributing stations, P-S pair count, azimuthal gap, fit metrics |
| `csv/phase_picks.csv` | One row per pick: absolute arrival time, travel time, distance, azimuth, probability |
| `csv/qc_event_report.csv` | Every QC metric for every candidate event, including those rejected, with the reason |

### Ready for hypocentre location

| File | Purpose |
| --- | --- |
| `nonlinloc/picks/<YYYYMMDD-HHMM-SS>.pick` | One NonLinLoc phase file per event |
| `nonlinloc/all_events.obs` | Combined archive of all events |
| `nonlinloc/stations_GTSRCE.txt` | Station block for the NonLinLoc control file |

Pick uncertainties are written as 0.10 s for P and 0.20 s for S, reflecting the fact
that S arrivals are genuinely harder to identify, so the location algorithm weights them
accordingly.

### Visual verification — the final arbiter

Every event, accepted or rejected, receives its own folder containing four items: the
**waveform plot** with the associated P and S picks marked, the **pseudo-distance plot**,
the **Wadati diagram**, and the pick table.

```
events/ev000013/                    ← passed QC
    waveform_ev000013.png
    pseudo_distance_ev000013.png
    wadati_ev000013.png
    picks_ev000013.csv

rejected/ev000009/                  ← rejected, with the reason in the title
    ...
```

The waveform figure is the decisive check. Stations are ordered by arrival time, with
the nearest at the bottom, so a real earthquake shows a clean **moveout** — the arrival
sweeping progressively later with distance. Amplitude is normalised within the display
window only, so distant, weaker traces remain readable.

Rejected events are plotted too. A quality-control system that cannot be audited is not
trustworthy, and reviewing the rejected folder is how the thresholds themselves get
validated.

---

## 7. Design principles

**Physical thresholds, not statistical ones.** Every rejection criterion has physical
units and physical meaning. A percentile threshold discards a fixed fraction of data
whatever its quality; a physical threshold rejects nothing when the data are sound.

**Independent tests.** The five QC stages probe different properties — structure,
distance, geometry, velocity, timing. An event passing all five is unlikely to be a
coincidence, because a coincidence would have to satisfy five unrelated constraints.

**No fabricated measurements.** Missing observations are never estimated from the
quantity being solved for. A missing S arrival stays missing.

**Derived, not tuned.** Every geometric parameter follows from the station layout, so
the pipeline transfers to another network without re-tuning.

**Visual inspection as the final authority.** Automated tests narrow the candidates;
the waveform plots decide. Every event is rendered, including the rejections.
