# SG-Nav on RescueBench

This baseline uses the upstream [SG-Nav](https://github.com/bagh2178/SG-Nav) model and its checkpoints. RescueBench supplies the adapter, launcher, and the four modified SG-Nav source files in `overlay/`. The overlay originated from the earlier RescueBench SG-Nav experiment and is released under the upstream [MIT license](UPSTREAM_LICENSE). The source checkout used for the adaptation was upstream commit `d56863c96dea311aaa67fb0d39a1a8ccc3f0487f`.

Install SG-Nav and its GLIP, SAM, GroundingDINO, Habitat, and checkpoint dependencies using the upstream instructions. The large model weights are not included here. Then apply the RescueBench overlay to a separate SG-Nav checkout:

```bash
git clone https://github.com/bagh2178/SG-Nav.git baseline_model/SG-Nav
git -C baseline_model/SG-Nav checkout d56863c96dea311aaa67fb0d39a1a8ccc3f0487f
cp benchmark/sgnav/overlay/SG_Nav_unreal.py baseline_model/SG-Nav/
cp benchmark/sgnav/overlay/scenegraph.py baseline_model/SG-Nav/
cp benchmark/sgnav/overlay/utils/utils_glip*.py baseline_model/SG-Nav/utils/
export SGNAV_ROOT="$PWD/baseline_model/SG-Nav"
export UnrealEnv=/absolute/path/to/UnrealEnv
# Set DASHSCOPE_API_KEY in the process environment if using the scene-graph LLM.
cd benchmark
python run_sgnav.py --levels 0 --episodes 1
```

For a single real test point before a full run, use `python sgnav/smoke_one.py --level 0 --point-id 0`. If the model files are already cached and the server cannot reach Hugging Face, set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` before launching. The smoke test writes one episode result under `benchmark_results/sgnav_smoke/`.

The adapter can also be selected with `python rescue_benchmark.py --model sgnav`. Its profile selects real 640 × 480 RGB-D observations and benchmark-managed carry/drop interactions. `--sgnav-root` and `--sgnav-config` override the default checkout and Habitat YAML paths. The launcher accepts the shared benchmark options, including `--output`, `--levels`, `--episodes`, and `--no-collision`.

## Exact adaptation choices

| Question | Implementation |
| --- | --- |
| Person / injured person | Phase 1 selects `person`; phase 2 selects `car` as a visual proxy for the ambulance/stretcher destination. This is an approximation, not a new learned stretcher class. |
| 21-class priors | `obj.npy` remains 21 × 21 and `room.npy` remains 21 × 9. For `person`/`car`, `SG_Nav_unreal.py` uses zero vectors instead of indexing a nonexistent row. No new co-occurrence statistics were learned. |
| Detection and scene graph | The GLIP vocabulary and scene-graph handling were adapted for `person` and `car`; see `overlay/utils/` and `overlay/scenegraph.py`. The default is the indoor vocabulary (`openworld=0`); `--sgnav-openworld` opts into the alternate vocabulary. |
| `TURN_RIGHT_2` | SG-Nav action ID 6 maps directly to RescueBench continuous action `((60, 0), 0, 0)`, a 60-unit right-turn command. It does not rely on a Habitat v1 `ActionSpec` during RescueBench execution. |
| STOP and rescue interaction | SG-Nav STOP maps to zero motion. In the default active benchmark mode, the common RescueBench state machine uses the simulator's target poses and distance gates to issue carry/drop actions. This is environment-assisted interaction, and should be reported when comparing with fully autonomous interaction baselines. |

RGB-D and pose come from the live Gym environment. The adapter converts depth from centimetres to metres, pose position from centimetres to metres, and maps SG-Nav movement to the shared mixed action interface. It raises an error if depth or pose is missing instead of fabricating input. The source overlay reads `DASHSCOPE_API_KEY` from the environment and contains no API credential.

For faithful default scene-graph relation scoring, provide a valid `DASHSCOPE_API_KEY`; without it, relation inference is unavailable and the resulting scores should not be treated as a full SG-Nav reproduction. The original SG-Nav repository and its checkpoint licenses and installation instructions continue to apply. Run this baseline in its own Python environment if its CUDA/Habitat dependencies conflict with other baselines.
