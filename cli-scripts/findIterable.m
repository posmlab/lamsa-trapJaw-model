%% helper function to detect iterable parameters
function iterable = findIterable(config)
    iterable = cell(0,0);
    params = {config.parameters};

    for component = 1:5
        component_params = fields(params{component});
        for i = 1:length(component_params)
            val = getfield(params{component}, component_params{i});
            if length(val) == 3
                iterable = [iterable; {component, component_params{i}, 1, linspace(val(1), val(2), val(3))}];
            elseif length(val) ~= 1
                error('inputted parameter has wrong length. must be scalar or length 3 array');
            end
        end
    end
end