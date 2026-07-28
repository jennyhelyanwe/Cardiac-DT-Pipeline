# Cardiac-DT-Pipeline

Orchestrator repo linking the stages of the maternal cardiac digital twin pipeline:
DICOM → surface mesh → volumetric mesh → EP inference → electromechanics simulation.

This repo does not contain pipeline code itself — each stage lives in its own
repo, included here as a git submodule.

## Pipeline stages
| Submodule                          | Role                                                    |
|-------------------------------------|----------------------------------------------------------|
| `biv-me`                            | DICOM → surface mesh fitting tool                        |
| `cardiac_dicom_to_surface_mesh`     | Colab glue code calling biv-me                           |
| `cardiac_surface_to_volumetric_mesh`| Surface mesh → volumetric mesh (TetGen)                  |
| `sim-based-inf-clin`                | Volumetric mesh → EP characteristic inference (SMC-ABC)  |
| `geometry-modulation`               | Chamber anatomy modulation → new simulation-ready meshes |
| `pregnancy-simcardems-sims`         | Simcardems runs across baseline + variant meshes × trimesters; ECG/PV biomarker sensitivity analysis |
| `simcardems` (branch: `biv-dev`)    | Electromechanics simulation                              |

## Provenance

- `biv-me` — from [UOA-Heart-Mechanics-Research/biv-me](https://github.com/UOA-Heart-Mechanics-Research/biv-me)
- `simcardems` — fork of [ComputationalPhysiology/simcardems](https://github.com/ComputationalPhysiology/simcardems), working branch `biv-dev`
- `sim-based-inf-clin` — fork of [JamesAlecColeman/sim-based-inf-clin](https://github.com/JamesAlecColeman/sim-based-inf-clin), working branch `popht-maternal-personalisation`


## Getting started

Clone with submodules included:
```bash
git clone --recurse-submodules https://github.com/jennyhelyanwe/Cardiac-DT-Pipeline.git
```

If already cloned without that flag:
```bash
git submodule update --init --recursive
```

## Working on a submodule

Each submodule is a normal, independent git repo. `cd` in and commit/push as usual:
```bash
cd simcardems
git checkout biv-dev
# make changes
git add . && git commit -m "..." && git push
```

**After pushing inside a submodule, bump the pointer in this repo** — this is the
one step that's easy to forget:
```bash
cd ..
git add simcardems
git commit -m "Bump simcardems to latest"
git push
```
`git status` here will flag any submodule whose commit has moved but hasn't been
bumped yet, worth checking before ending a work session.

## Notes

- `simcardems` tracks the `biv-dev` branch and `sim-based-inf-clin` tracks the `popht-maternal-personalisation` branch (set via `.gitmodules`); other submodules
  track their default branch unless noted otherwise.
- Data files are kept out of these repos — see individual submodule `.gitignore`s.
- Standardized data contract between stages (subject/variant naming,
  provenance tracking) planned, watch this space. 
