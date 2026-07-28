# Luna — MPC Guidance & Control: Presentation Outline

Source material: [`../guided_parachutes.md`](../guided_parachutes.md), [`../README.md`](../README.md).

Build the deck with:

```bash
pip install python-pptx
python presentation/build_deck.py
```

Output: `presentation/Luna_MPC_Parafoil_Recovery.pptx` (16:9, fully editable text
boxes and tables — no images, so it imports cleanly into PowerPoint, Google
Slides, Keynote, and Canva).

## Slide plan

| # | Slide | Purpose |
|---|-------|---------|
| 1 | Title | Project name, subtitle, framing |
| 2 | Mission | What the system must do |
| 3 | Two axes of success | Theoretical rigor vs practical trustworthiness |
| 4 | Parallel tracks | Primary MPC track and fallback guidance track |
| 5 | Technical domains | Scope of the work |
| 6 | GNC stack | Sensing → estimation → guidance → control → actuation |
| 7 | Why MPC | Receding-horizon feedback as the source of robustness |
| 8 | Current MPC formulation | Nonlinear receding horizon, `fmincon`/SQP |
| 9 | Wind estimation | GPS ground velocity vs predicted air velocity, low-pass filtered |
| 10 | Three failure modes | Bias, constraint violation, terminal model reliance + mitigations |
| 11 | Solver tradeoff | SQP vs LTV-QP + OSQP for embedded flight |
| 12 | Fallback guidance | Phased waypoint guidance law |
| 13 | Mechanical safety | Servo out of the riser load path; graceful failure |
| 14 | Roadmap | Solver migration, flare, successive convexification, model upgrades |
| 15 | Tools and code map | MATLAB/OSQP roles, key files |
| 16 | Takeaways | Design principles worth carrying forward |

## Editing notes

- Colours and fonts live in `THEME`, `FONT_HEAD`, `FONT_BODY`, and `FONT_MONO` at
  the top of `build_deck.py`; copy lives in the per-slide `slide_*` functions.
- `bullets()` and `card()` self-size from estimated wrapped text height
  (`wrapped_height`), so editing copy will not push text off the slide.
- Re-running the script overwrites the `.pptx`. Edit the script for structural
  changes and the `.pptx` directly for one-off polish.

## Preview / export

Render to PDF or PNG without PowerPoint:

```bash
soffice --headless --convert-to pdf --outdir /tmp presentation/Luna_MPC_Parafoil_Recovery.pptx
pdftoppm -png -r 110 /tmp/Luna_MPC_Parafoil_Recovery.pdf /tmp/slide
```

To use it in Canva: **Create a design → Import file** and upload the `.pptx`.
Text boxes, tables, and shapes stay individually editable after import.
