%% helper function to create components from run parameters
function components = initializeComponents(component_config, run_params)
    component_names = {component_config.name};
    components = cell(1,length(component_config));
    for component = 1:length(component_config)
        component_params = struct2cell(run_params{component});
        components{component} = feval(component_names{component}, component_params{:});
    end
end