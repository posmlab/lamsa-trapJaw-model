%% helper function to set run parameters
function run = setRunParams(config, iterable)
    run = {config.parameters};
    num_iterable = size(iterable, 1);
    for param = 1:num_iterable
        run{iterable{param, 1}} = setfield(run{iterable{param, 1}}, iterable{param, 2}, iterable{param, 4}(iterable{param, 3}));
    end
end