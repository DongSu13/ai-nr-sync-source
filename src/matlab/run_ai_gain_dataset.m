function run_ai_gain_dataset(configPath, outputPath, seedSelection)
if nargin < 1, configPath = fullfile('configs','stage4_ai_gain.json'); end
if nargin < 2, outputPath = fullfile('outputs','ai_gain','candidates.jsonl'); end
cfg = jsondecode(fileread(configPath));
if nargin < 3, seedSelection = cfg.seeds; fileMode = 'w'; else, fileMode = 'a'; end
outDir = fileparts(outputPath); if ~isfolder(outDir), mkdir(outDir); end
fid = fopen(outputPath,fileMode); assert(fid ~= -1); cleanup = onCleanup(@() fclose(fid));
for seed = reshape(seedSelection,1,[])
    scenario = struct('seed',seed,'scs_khz',cfg.scs_khz,'nsize_grid',cfg.nsize_grid,...
        'snr_db',cfg.snr_db(mod(seed,numel(cfg.snr_db))+1),'cfo_search_hz',cfg.cfo_search_hz,...
        'channel',cfg.channels{mod(seed,numel(cfg.channels))+1});
    scenario.users = [struct('pci',42,'timing_offset_samples',191,'cfo_hz',300,'power_db',0),...
        struct('pci',517,'timing_offset_samples',450,'cfo_hz',-200,'power_db',cfg.power_db(mod(seed,numel(cfg.power_db))+1))];
    result = run_two_user_sic(scenario,'');
    for k=1:numel(result.ai_candidates)
        c=result.ai_candidates(k);
        f=struct('pss_metric',c.pss_metric,'pss_peak_to_sidelobe',c.pss_peak_to_sidelobe,...
          'sss_metric',c.sss_metric,'pss_sss_consistency',c.pss_sss_consistency,'correlation_metric',c.correlation_metric,...
          'abs_cfo_hz',abs(c.cfo_hz),'dmrs_gain_abs',c.dmrs_gain_abs,'dmrs_gain_variance',c.dmrs_gain_variance,...
          'llr_mean_abs',c.llr_mean_abs,'received_power',result.metrics.pre_sic_power,'residual_power',result.metrics.post_sic_residual_power);
        label=double(c.pbch_crc_ok && any(c.pci==[scenario.users.pci]));
        row=struct('split_seed',seed,'channel',scenario.channel,'snr_db',scenario.snr_db,'features',f,'label',label);
        fprintf(fid,'%s\n',jsonencode(row));
    end
end
end
