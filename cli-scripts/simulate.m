%% helper function to run LaMSA and MDA simulations + report values for given components
function [lamsa_met, mda_met] = simulate(components, mets)

    [lamsa_sol, lamsa_transitions] = solve_lamsa(components{4}, components{5}, components{1}, components{2}, components{3});
    lamsa_met = get_metrics(lamsa_sol, lamsa_transitions, components{1}, mets);

    [mda_sol, mda_transitions] = solve_direct_actuation(components{4}, components{1});
    mda_met = get_metrics(mda_sol, mda_transitions, components{1}, mets);

end