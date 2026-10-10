function run_ai_dataset_generation(configPath, outputPath, seedSelection)
%RUN_AI_DATASET_GENERATION Build labeled, seed-grouped records from NR runs.
if nargin < 1
    configPath = fullfile('configs', 'stage3_ai_dataset.json');
end
if nargin < 2
    outputPath = fullfile('outputs', 'ai', 'candidates.jsonl');
end
cfg = jsondecode(fileread(configPath));
if nargin < 3
    seedSelection = cfg.seeds;
    fileMode = 'w';
else
    fileMode = 'a';
end
outDir = fileparts(outputPath);
if ~isfolder(outDir)
    mkdir(outDir);
end
fid = fopen(outputPath, fileMode);
assert(fid ~= -1, 'Cannot open AI dataset output path.');
cleanup = onCleanup(@() fclose(fid));
for seed = reshape(seedSelection, 1, [])
    for snr = reshape(cfg.snr_db, 1, [])
        scenario = struct();
        scenario.seed = seed;
        scenario.scs_khz = cfg.scs_khz;
        scenario.nsize_grid = cfg.nsize_grid;
        scenario.snr_db = snr;
        scenario.cfo_search_hz = cfg.cfo_search_hz;
        scenario.users = [ ...
            struct('pci', 42, 'timing_offset_samples', 191, 'cfo_hz', 300, 'power_db', 0), ...
            struct('pci', 517, 'timing_offset_samples', 450, 'cfo_hz', -200, ...
                'power_db', cfg.power_db(mod(seed + snr, numel(cfg.power_db)) + 1))];
        result = run_two_user_sic(scenario, '');
        for record = result.ai_records
            fprintf(fid, '%s\n', jsonencode(record));
        end
    end
end
end
