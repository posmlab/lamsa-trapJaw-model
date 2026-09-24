%% RotatingMass class definition
%
% this type of mass allows large angle lever arm rotations for variable EMA
%
% arguments:
%     V_load - volume of rod-like load
%     density - density of lever arm
%     L1    - segment of the lever arm closest to the spring
%     L2    - segment of the lever arm furthest from the spring
%     x_rest- rest length of the spring
% min # arguments = 1

classdef RotatingMass < Mass
    
    properties
        L1, theta_0, theta_f, x_p
    end
    
    methods (Static)
        function parameters = parameters()
            parameters = ["volume of load" "density" "L1" "L2" "theta initial" "theta final" "pivot x";
                "0.01" "0" "0.001" "0.001" "0" "0" "0";
                "0" "0" "0" "0" "-1.57" "-1.57" "-Inf";
                "Inf" "Inf" "Inf" "Inf" "1.57" "1.57" "Inf"];
        end
    end
    
    methods
        
        % constructor
        function obj = RotatingMass(V_load, density, L1, L2, theta_0, theta_f)

            % model
            EMA = L1/L2;
            mass = V_load * density * ( (1+1/EMA)^2 + 3*(1/EMA-1)^2 ) / 12;
            EMA = @(y) EMA;
            
            % call parent constructor
            obj = obj@Mass(mass, EMA);
            obj.L1 = L1;
            obj.theta_0 = theta_0;
            obj.theta_f = theta_f;
            obj.x_p = x_p;
        end 
    end
end
