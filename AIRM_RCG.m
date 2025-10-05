
function [C] = AIRM_RCG(X_train, labels, C0)
    
    X = X_train;
    C = C0;

    N = length(labels);
    d = size(X{1},1);

    spd_manifold = powermanifold(sympositivedefinitefactory(d), 4);
    spd_problem.M = spd_manifold; 
    spd_problem.cost = @spd_objective;
    spd_problem.egrad = @spd_gradient; 
    
    function obj = objective(C)

        obj = 0;

        for i=1:N
            d_he = + 2*AIRM(X{i}, C{1}) ...
                + 1*AIRM(X{i}, C{2}) ...
                + 1*AIRM(X{i}, C{3});

            d_ne = - 2*AIRM(X{i}, C{4});

            label = labels(i);

            obj = obj ...
                + label * max(0, d_ne - d_he) ...
                + (1-label) * max(0, d_he - d_ne);
        end
    end

    function obj = spd_objective(C) 
        obj = objective(C);
    end

    function gradC = spd_gradient(C)

        gradC = cell(4,1);
        [gradC{:}] = deal(0);

        for i=1:N
            d_he = + 2*AIRM(X{i}, C{1}) ...
                + 1*AIRM(X{i}, C{2}) ...
                + 1*AIRM(X{i}, C{3});

            d_ne = - 2*AIRM(X{i}, C{4});

            label = labels(i);

            if label == 1 && d_ne > d_he

                gradC{4} = gradC{4} + 2 * AIRM_grad(X{i}, C{4});

            elseif label == 0 && d_he > d_ne

                gradC{1} = gradC{1} + 2 * AIRM_grad(X{i}, C{1});
                gradC{2} = gradC{2} + 1 * AIRM_grad(X{i}, C{2});
                gradC{3} = gradC{3} + 1 * AIRM_grad(X{i}, C{3});

            end
            
            gradB{t} = gradB{t} + real(((similar(n)+dissimilar(n))*mv12)*(gradBit-gradBjt));
        
        end

    end

    opts.verbosity = 0;

    for iter = 1:5

        C = conjugategradient(spd_problem, C, opts);

        obj = objective(C);
        fprintf('\titer = %d : Loss = %0.2f\n', iter, obj)

    end
end

