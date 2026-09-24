addpath(genpath(fullfile(pwd,'..')));
RUN_PATH = input('path to simulation run directory: ', 's');
COMPONENTS_PATH = fullfile(RUN_PATH, 'components-config.json');
METRICS_PATH = fullfile(RUN_PATH, 'metrics-config.json');

component_order = {'load', 'latch', 'spring', 'load motor', 'latch motor'};
component_config = jsondecode(fileread(COMPONENTS_PATH));

if length(component_config) ~= length(component_order)
    error('components-config.json must describe all 5 components');
end

if ~isequal(component_order, {component_config.component})
    error('components-config.json must order components as load, latch, spring, load motor, latch motor');
end


iterable_params = findIterable(component_config);
num_iterable = size(iterable_params, 1);
num_runs = 1;
for param = 1:num_iterable
    num_runs = num_runs .* size(iterable_params{param, 4}, 2);
end


metrics = sort(jsondecode(fileread(METRICS_PATH)));
num_metrics = length(metrics);

sim_output = cell(1, num_iterable + 2*length(metrics));
for param = 1:num_iterable
    sim_output{1,param} = iterable_params{param, 2};
end
for metric = 1:num_metrics
    sim_output{1, num_iterable + metric} = strcat('lamsa-', metrics{metric});
    sim_output{1, num_iterable + num_metrics + metric} = strcat('mda-', metrics{metric});
end


for run = 1:num_runs
    run_params = setRunParams(component_config, iterable_params);
    components = initializeComponents(component_config, run_params);

    [lamsa_metrics, mda_metrics] = simulate(components, metrics);
    run_output = cell(1, num_iterable);
    for param = 1:num_iterable
        run_output{1,param} = getfield(run_params{iterable_params{param, 1}}, iterable_params{param, 2});
    end
    run_output = [run_output, values(lamsa_metrics)];
    run_output = [run_output, values(mda_metrics)];

    sim_output = [sim_output; run_output];

    for param = 1:num_iterable
        if iterable_params{param, 3} < size(iterable_params{param, 4}, 2)
            iterable_params{param, 3} = iterable_params{param, 3} + 1;
            break;
        end
        iterable_params{param, 3} = 1;
    end
end

writetable(cell2table(sim_output(2:end,:), 'VariableNames', sim_output(1,:)), fullfile(RUN_PATH, 'simulation_outputs.csv'));