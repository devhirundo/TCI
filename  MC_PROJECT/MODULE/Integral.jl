module Integral

    struct MeshFunction
        f::Function
        domain::Tuple{Real, Real}
        mesh::Int
    end

    function Base.getindex(mf::MeshFunction, idx::Int)
        minval, maxval = mf.domain
        grid = mf.mesh

        coord = minval + (idx - 1) * (maxval - minval) / (grid - 1)
        return mf.f(coord)
    end

    function sum_all_elements(mf::MeshFunction)
        minval, maxval = mf.domain
        grid = mf.mesh
        dv = (maxval - minval) / (grid - 1)
        sumVal = 0.0
        for idx in 1:grid
            sumVal += mf[idx]
        end
        return sumVal * dv
    end


    function get_Z(mf::MeshFunction)
        minval, maxval = mf.domain
        grid = mf.mesh
        dv = (maxval - minval) / (grid - 1)
        sumVal = 0.0
        for idx in 1:grid
            sumVal += abs(mf[idx])
        end
        return sumVal * dv
    end


    function sum_sign_with_mcmc(mf::MeshFunction, step::Int, initstep::Int)
        grid = mf.mesh
        j::Int = rand(1:grid)
        now = mf[j]
        k = max(1, round(Int, grid / 10))

        #Burn in
        for _ in 1:initstep
            jump = rand(-k:k)
            if 1 <= (j + jump) <= grid
                j += jump
                prp = mf[j]
                if rand() < abs(prp / now)
                    now = prp
                else
                    j -= jump
                end
            end
        end

        # Main Sampling
        sumSgn = 0.0
        for _ in 1:step
            jump = rand(-k:k)
            
            if (j + jump) < 1 || (j + jump) > grid
                sumSgn += sign(now)
            else
                j += jump
                prp = mf[j]
                accRatio = abs(prp / now)
                
                if rand() < accRatio
                    now = prp
                    sumSgn += sign(now)
                else
                    sumSgn += sign(now)
                    j -= jump
                end
            end
        end
        return sumSgn / step
    end
end