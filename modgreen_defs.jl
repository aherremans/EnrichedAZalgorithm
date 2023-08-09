## This code is copied from the package SingularIntegralEquations.jl
# It isolates all code required to evaluate the gravity Helmholtz Green's function.

# Syntax:
#
#
# x0 = [0.0; 0]
# y0 = [0; 10.0]
# E = 5.0
# lhelmfs(x0[1]+im*x0[2], y0[1]+im*y0[2], 5.0)
# 0.03327289483292037 + 0.041618616932094635im

# some useful math constants.
const M_PI_3 = 1.047197551196598
const M_PI_2 = 1.5707963267948966
const M_PI = 3.1415926535897932384
const M_1_PI = 0.31830988618379067
const M_1_4PI = 7.957747154594767e-02
const M_PI_4 = 0.785398163397448309

const M_F_1 = M_1_PI + 0.5
const M_F_2 = M_PI_4 - 0.5
const M_G_1 = M_1_PI + 1.0 / 6.0
const M_G_2 = M_PI_4 / 3.0 - 0.5

const THIRD = 0.333333333333333333
const ZIM = 0.0im

const W = 0.363630003348128         # precomputed constant for finding stationary points
const V = -0.534877842831614        # precomputed constant for finding stationary points
const W2 = 0.5*0.363630003348128    # precomputed constant for finding stationary points
const V4 = -0.25*0.534877842831614  # precomputed constant for finding stationary points
const jump_ratio = 1.3              # when finding endpoints, increase distance from, 1.3
	                                # stationary point by this ratio

const MAXNQUAD = 3000               # maximum allowed number of quad points

# Allocation
const gam = Vector{Complex{Float64}}(undef,MAXNQUAD)
const gamp = Vector{Complex{Float64}}(undef,MAXNQUAD)
const integ = Vector{Complex{Float64}}(undef,MAXNQUAD)
const integx = Vector{Complex{Float64}}(undef,MAXNQUAD)
const integy = Vector{Complex{Float64}}(undef,MAXNQUAD)
const ts = Vector{Float64}(undef,MAXNQUAD)
const ws = Vector{Float64}(undef,MAXNQUAD)

# locate_minimum allocation

const n_its = 2
const n_t = 20
const lm_ts = Vector{Float64}(undef,n_t)
const lm_gam = Vector{Complex{Float64}}(undef,n_t)
const lm_integ = Vector{Complex{Float64}}(undef,n_t)
const lm_integx = Vector{Complex{Float64}}(undef,n_t)
const lm_integy = Vector{Complex{Float64}}(undef,n_t)
const test_min = Vector{Float64}(undef,n_t)

# find_endpoints allocation

const fe_gam = Vector{Complex{Float64}}(undef,1)
const fe_u = Vector{Complex{Float64}}(undef,1)
const fe_ux = Vector{Complex{Float64}}(undef,1)
const fe_uy = Vector{Complex{Float64}}(undef,1)

# default numerical params, and how they scale with h (when h<0 triggers it):
const MAXH_STD = 0.05           # max h spacing on real axis for quad-nodes meth=1, .03
const MINSADDLEN_STD = 43       # min nodes per saddle pt: 40
const MAXH_OVER_H = 0.13        # ratio in mode when h scales things: 0.14
const MINSADDLEN_TIMES_H = 15   # ratio in mode when 1/h scales things: 14

function integrand!(u::Vector{Complex{Float64}}, ux::Vector{Complex{Float64}}, uy::Vector{Complex{Float64}}, s::Vector{Complex{Float64}}, a::Float64, b::Float64, x::Float64, y::Float64, derivs::Bool, n::Int)
    @simd for i=1:n
        # Precalculations
        @inbounds es = exp(s[i])
        ies = inv(es)
        @inbounds u[i] = cis(a * ies + b * es - es * es * es / 12.0)

        if derivs
            # calculate prefactors for exponentials in order to do derivatives
            c,d = 0.5im*ies,0.5im*es
            # x deriv computation
            @inbounds ux[i] = (-c*x)*u[i]
            # y deriv computation
            @inbounds uy[i] = (-c*y+d)*u[i]
        else
            # make sure that ux, uy aren't going to give us anything funny (or >emach)
            @inbounds ux[i] = ZIM
            @inbounds uy[i] = ZIM
        end
    end
end

# assume if length n not given that vectors are of length 1.

integrand!(u::Vector{Complex{Float64}}, ux::Vector{Complex{Float64}}, uy::Vector{Complex{Float64}}, s::Vector{Complex{Float64}}, a::Float64, b::Float64, x::Float64, y::Float64, derivs::Bool) = integrand!(u, ux, uy, s, a, b, x, y, derivs, 1)

function gammaforbidden!(gam::Vector{Complex{Float64}}, gamp::Vector{Complex{Float64}}, s0::Complex{Float64}, s::Vector{Float64}, n::Int)
    res,ims = reim(s0)
    @simd for i=1:n
        d = s[i] - res
        # region 1 (deep forbidden)
        if ims ≤ -M_PI_3
            @inbounds gam[i] = complex(s[i],THIRD*atan(d) - M_PI_3)
            @inbounds gamp[i] = complex(1.0,THIRD/(1+d*d))
        else
            ims = imshack(res,ims)
            d2 = d^2
            ex = exp(-d2)
            g0 = THIRD*(atan(d)-M_PI)
            @inbounds gam[i] = complex(s[i],ims + (g0 - ims)*(1.0-ex))
            @inbounds gamp[i] = complex(1.0,2.0*(g0 - ims)*d*ex + THIRD / (1.0 + d2) * (1.0-ex))
        end
    end
end

# assume if length n not given that vectors are of length 1 (with a slightly different signature)

function gammaforbidden!(gam::Vector{Complex{Float64}}, s0::Complex{Float64}, s::Float64)
    res,ims = reim(s0)
    d = s - res
    # region 1 (deep forbidden)
    if ims ≤ -M_PI_3
        gam[1] = complex(s,THIRD*atan(d) - M_PI_3)
    else
        ims = imshack(res,ims)
        d2 = d^2
        ex = exp(-d2)
        g0 = THIRD*(atan(d)-M_PI)
        gam[1] = complex(s,ims + (g0 - ims)*(1.0-ex))
    end
end

function imshack(res::Float64,ims::Float64)
    # hack to move ctr just below coalescing saddle at high E. Note exp(res) ~ sqrt(E).
    if ims > -0.1 ims -= max(0.0,min(0.7exp(-res),0.1)) end
    return ims
end


function contour!(gam::Vector{Complex{Float64}}, gamp::Vector{Complex{Float64}}, a::Float64, b::Float64, ts::Vector{Float64}, n::Int)
    # Complex precalculations for stationary points
    temp1 = sqrt(b^2-a+ZIM)
    temp2 = sqrt(a)
    # Relevant stationary points of the integrand
    st1 = 0.5log(2.0(b + temp1))
    st3 = 0.5log(2.0(b - temp1))
    # swap so st3 has smaller real part for contour only
    if real(st3) > real(st1) st1,st3 = st3,st1 end
    # hack to deal with brach of log
    if imag(st3) ≥ M_PI_2 st3 -= im*M_PI end
    # construct gam, gamp
    if b ≤ temp2
        gammaforbidden!(gam,gamp,st3,ts,n)
    else
        # hack to move ctr just below coalescing saddle at high E:
        # note exp(re(s)) ~ sqrt(E). See also gammaforbidden!()
        imsh = 0.0
        if abs(real(st1)-real(st3)) < 0.1
            imsh = imshack(real(st1),imsh)
        end
        @simd for i=1:n
            @inbounds temp1c=2.0(ts[i] - real(st3)) + W
            @inbounds temp2c=-4.0(ts[i] - real(st1)) + V

            f = M_F_1 * atan(temp1c) - M_F_2
            fp = 2.0M_F_1 / (1.0 + temp1c^2)

            g = M_G_1 * atan(temp2c) - M_G_2
            gp = -4 * M_G_1 / (1.0 + temp2c^2)

            @inbounds gam[i] = complex(ts[i],f*g+imsh)
            @inbounds gamp[i] = complex(1.0,fp*g+f*gp)
        end
    end
end


function locate_minimum(st1::Complex{Float64}, st3::Complex{Float64}, a::Float64, b::Float64, x::Float64, y::Float64, derivs::Bool)
    rst1,rst3 = real(st1),real(st3)
    t1,t2 = rst1,rst3
    tmin,fmin = 0.0,0.0
    # number of loops
    for k=1:n_its
        linspace!(lm_ts,t1,t2,n_t)
        # Precalculations
        for i=1:n_t
            temp1c=2.0(lm_ts[i] - rst3) + W
            temp2c=-4.0(lm_ts[i] - rst1) + V
            f = M_F_1 * atan(temp1c) - M_F_2
            g = M_G_1 * atan(temp2c) - M_G_2
            lm_gam[i] = complex(lm_ts[i],f*g)
        end
        integrand!(lm_integ,lm_integx,lm_integy,lm_gam,a,b,x,y,derivs,n_t)
        # function to be minimized: square of two norm of u, ux, uy
        for i=1:n_t
            test_min[i] = abs2(lm_integ[i])+abs2(lm_integx[i])+abs2(lm_integy[i])
        end

        fmin,idx = findmin(test_min)
        if idx == 1 || idx == n_t || k == n_its
            tmin = lm_ts[idx]
        else
            t1,t2 = lm_ts[idx-1],lm_ts[idx+1]
        end
    end
    return tmin,fmin
end



function find_endpoints(a::Float64, b::Float64, x::Float64, y::Float64, derivs::Bool)
    ϵ = 1e-14
    temp1 = sqrt(b^2-a+ZIM)
    tm = sqrt(2.0(b-temp1))
    tp = sqrt(2.0(b+temp1))
    st1 = log(tp)
    st3 = log(tm)
    if imag(st3) ≥ M_PI_2 st3 -= im*M_PI end

    if b ≤ sqrt(a)
        c1 = real(st1)
        stdist = 0.5abs(c1)
        if stdist == 0 stdist = 1.0 end

        # We start with finding lm1
        t = c1 - stdist
        gammaforbidden!(fe_gam, st3, t)
        dlm1 = 1.0
        integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
        if abs(fe_u[1]) > ϵ || abs(fe_ux[1]) > ϵ || abs(fe_uy[1]) > ϵ
            while abs(fe_u[1]) > ϵ || abs(fe_ux[1]) > ϵ || abs(fe_uy[1]) > ϵ
                dlm1 *= jump_ratio
                t = c1 - dlm1*stdist
                gammaforbidden!(fe_gam, st3, t)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
        else
            while abs(fe_u[1]) < ϵ && abs(fe_ux[1]) < ϵ && abs(fe_uy[1]) < ϵ && dlm1 > ϵ
                dlm1 /= jump_ratio
                t = c1 - dlm1*stdist
                gammaforbidden!(fe_gam, st3, t)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
            if dlm1 < ϵ dlm1 = 0.0 end
            dlm1 *= jump_ratio
        end
        lm1 = c1 - dlm1*stdist

        # lp1 comes next
        t = c1 + stdist
        gammaforbidden!(fe_gam, st3, t)
        dlm1 = 1.0
        integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
        if abs(fe_u[1]) < ϵ && abs(fe_ux[1]) < ϵ && abs(fe_uy[1]) < ϵ
            while abs(fe_u[1]) < ϵ && abs(fe_ux[1]) < ϵ && abs(fe_uy[1]) < ϵ && dlm1 > ϵ
                dlm1 /= jump_ratio
                t = c1 + dlm1*stdist
                gammaforbidden!(fe_gam, st3, t)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
            if dlm1 < ϵ dlm1 = 0.0 end
            dlm1 *= jump_ratio
        else
            while abs(fe_u[1]) > ϵ || abs(fe_ux[1]) > ϵ || abs(fe_uy[1]) > ϵ
                dlm1 *= jump_ratio
                t = c1 + dlm1*stdist
                gammaforbidden!(fe_gam, st3, t)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
        end
        lp1 = c1 + dlm1*stdist

        c2 = c1
        lp2 = 0.0
        lm2 = 0.0
    else
        c1,c2 = real(st1),real(st3)
        if c1 ≥ c2
            c1,c2 = c2,c1
        end
        tmin,fmin = locate_minimum(st1, st3, a, b, x, y, derivs)

        f2 = -real(st3) + W2
        g2 = -real(st1) - V4

        stdist = abs(c1-tmin)
        if stdist == 0 stdist = 1.0 end

        # We start with finding lm1
        t = c1 - stdist

        f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
        g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
        fe_gam[1] = complex(t,f*g)

        dlm1 = 1.0
        integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
        if abs(fe_u[1]) > ϵ || abs(fe_ux[1]) > ϵ || abs(fe_uy[1]) > ϵ
            while abs(fe_u[1]) > ϵ || abs(fe_ux[1]) > ϵ || abs(fe_uy[1]) > ϵ
                dlm1 *= jump_ratio
                t = c1 - dlm1*stdist
                f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
                g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
                fe_gam[1] = complex(t,f*g)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
        else
            while abs(fe_u[1]) < ϵ && abs(fe_ux[1]) < ϵ && abs(fe_uy[1]) < ϵ && dlm1 > ϵ
                dlm1 /= jump_ratio
                t = c1 - dlm1*stdist
                f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
                g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
                fe_gam[1] = complex(t,f*g)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
            if dlm1 < ϵ dlm1 = 0.0 end
            dlm1 *= jump_ratio
        end
        lm1 = c1 - dlm1*stdist

        # lp1 comes next
        t = c1 + stdist

        f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
        g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
        fe_gam[1] = complex(t,f*g)

        dlm1 = 1.0
        integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
        if abs(fe_u[1]) < ϵ && abs(fe_ux[1]) < ϵ && abs(fe_uy[1]) < ϵ
            while abs(fe_u[1]) < ϵ && abs(fe_ux[1]) < ϵ && abs(fe_uy[1]) < ϵ && dlm1 > ϵ
                dlm1 /= jump_ratio
                t = c1 + dlm1*stdist
                f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
                g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
                fe_gam[1] = complex(t,f*g)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
            if dlm1 < ϵ dlm1 = 0.0 end
            dlm1 *= jump_ratio
        else
            while abs(fe_u[1]) > ϵ || abs(fe_ux[1]) > ϵ || abs(fe_uy[1]) > ϵ
                dlm1 *= jump_ratio
                t = c1 + dlm1*stdist
                f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
                g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
                fe_gam[1] = complex(t,f*g)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
        end
        lp1 = c1 + dlm1*stdist

        #
        # Next up: lm2
        #
        stdist = abs(c2-tmin)
        if stdist == 0 stdist = 1.0 end

        t = c2 - stdist

        f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
        g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
        fe_gam[1] = complex(t,f*g)

        dlm1 = 1.0
        integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
        if abs(fe_u[1]) > ϵ || abs(fe_ux[1]) > ϵ || abs(fe_uy[1]) > ϵ
            while abs(fe_u[1]) > ϵ || abs(fe_ux[1]) > ϵ || abs(fe_uy[1]) > ϵ
                dlm1 *= jump_ratio
                t = c2 - dlm1*stdist
                f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
                g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
                fe_gam[1] = complex(t,f*g)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
        else
            while abs(fe_u[1]) < ϵ && abs(fe_ux[1]) < ϵ && abs(fe_uy[1]) < ϵ && dlm1 > ϵ
                dlm1 /= jump_ratio
                t = c2 - dlm1*stdist
                f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
                g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
                fe_gam[1] = complex(t,f*g)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
            if dlm1 < ϵ dlm1 = 0.0 end
            dlm1 *= jump_ratio
        end
        lm2 = c2 - dlm1*stdist

        # Last but not least: lp2
        t = c2 + stdist

        f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
        g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
        fe_gam[1] = complex(t,f*g)

        dlm1 = 1.0
        integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
        if abs(fe_u[1]) < ϵ && abs(fe_ux[1]) < ϵ && abs(fe_uy[1]) < ϵ
            while abs(fe_u[1]) < ϵ && abs(fe_ux[1]) < ϵ && abs(fe_uy[1]) < ϵ && dlm1 > ϵ
                dlm1 /= jump_ratio
                t = c2 + dlm1*stdist
                f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
                g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
                fe_gam[1] = complex(t,f*g)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
            if dlm1 < ϵ dlm1 = 0.0 end
            dlm1 *= jump_ratio
        else
            while abs(fe_u[1]) > ϵ || abs(fe_ux[1]) > ϵ || abs(fe_uy[1]) > ϵ
                dlm1 *= jump_ratio
                t = c2 + dlm1*stdist
                f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
                g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
                fe_gam[1] = complex(t,f*g)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
        end
        lp2 = c2 + dlm1*stdist

        # final cleanup
        if lm2 < c1
            if lm1 < lm2
                lm1 = lm2
            end
            lm2 = c2
        end
        if lp1 > c2
            if lp1 < lp2
                lp2 = lp1
            end
            lp1 = c1
        end
    end
    return lm1,lp1,lm2,lp2,c1,c2,tm,tp
end



function quad_nodes!(ts::Vector{Float64}, ws::Vector{Float64}, a::Float64, b::Float64, x::Float64, y::Float64, derivs::Bool, stdquad::Int, h::Float64, meth::Int)
    n = length(ts)
    @assert n == length(ws)

    maxh = MAXH_STD
    minsaddlen = MINSADDLEN_STD

    if meth == 1 && abs(h) ≤ 0.001
        warn("meth=1: option h must be > 1e-3 in size; setting to 0.3")
        h = 0.3
    end
    lm1,lp1,lm2,lp2,c1,c2,tm,tp = find_endpoints(a,b,x,y,derivs)
    sigm = 1.0/sqrt(abs(a/tm + b*tm - 0.75*tm*tm*tm))
    sigp = 1.0/sqrt(abs(a/tp + b*tp - 0.75*tp*tp*tp))

    if h < 0
        h = -h
        minsaddlen = ceil(Int,MINSADDLEN_TIMES_H/h)
        maxh = MAXH_OVER_H*h
    end

    n1,n2 = 0,0
    if ( lm2 == c2 || lp1 == c1 ) && c1 != c2
        # two maxima, no die-off in middle
        if meth == 0
            lga = log10(a)
            if lga < -0.5
                n2 = round(Int,2stdquad - stdquad*lga)
                if n2 > MAXNQUAD - stdquad n2 = MAXNQUAD - stdquad end
            else
                n2 = 2stdquad
            end
            dist1 = (lp2 - lm1) / ( n2 - 1)
            linspace!(ts,lm1,lp2,n2)
            ws[1:n2] = dist1*M_1_4PI
            n = n2
        elseif meth == 1
            h1 = min(h*min(sigm,sigp),maxh)
            n1 = ceil(Int,(lp2-lm1)/h1)
            if n1<minsaddlen n1=minsaddlen; h1=(lp2-lm1)/(n1-1) end
            if n1 > MAXNQUAD n1=MAXNQUAD end
            linspace!(ts,lm1,lm1+(n1-1)h1,n1)
            ws[1:n1] .= h1*M_1_4PI
            n = n1
        end
    elseif c1 != c2
        # two maxima, die-off in middle
        if meth == 0
            lga = log10(a)
            if lga < -0.5
                n2 = round(Int,stdquad - stdquad*lga)
                if n2 > MAXNQUAD - stdquad n2 = MAXNQUAD - stdquad end
            else
                n2 = stdquad
            end
            dist1 = (lp1-lm1)/(n2-1)
            dist2 = (lp2-lm)/(stdquad-1)
            linspace!(ts,lm1,lp1,n2)
            w[1:n2] = dist1
            n = n2 + stdquad
            linspace!(ts,lm2,lp2,n2,stdquad)
            ws[1+n2:n] .= dist2*M_1_4PI
        elseif meth == 1
            h1 = h*sigm
            if h1>maxh h1=maxh end
            n1 = ceil(Int,(lp1-lm1)/h1)
            if n1>MAXNQUAD/2 n1=div(MAXNQUAD,2) end
            linspace!(ts,lm1,lm1+(n1-1)h1,n1)
            ws[1:n1] .= h1*M_1_4PI
            h2 = h*sigp
            if h2>maxh h2=maxh end
            n2 = ceil(Int,(lp2-lm2)/h2)
            if n2>MAXNQUAD/2 n2=div(MAXNQUAD,2) end
            n = n1+n2
            linspace!(ts,lm2,lm2+(n2-1)h2,n1,n2)
            ws[1+n1:n] .= h2*M_1_4PI
        end
    else
        # one maximum
        if meth == 0
            n2 = stdquad
            lp2 = lp1
            dist1 = (lp2-lm1)/(n2-1)
            linspace!(ts,lm1,lp2,n2)
            ws[1:n2] .= dist1*M_1_4PI
            n = n2
        elseif meth == 1
            h1 = min(h*sigp,maxh)
            n1 = ceil(Int,(lp1-lm1)/h1)
            if n1<minsaddlen n1=minsaddlen; h1=(lp1-lm1)/(n1-1) end
            if n1>MAXNQUAD n1=MAXNQUAD end
            linspace!(ts,lm1,lm1+(n1-1)h1,n1)
            ws[1:n1] .= h1*M_1_4PI
            n = n1
        end
    end
    n
end

function linspace!(ts::Vector, start::Real, stop::Real, n::Int)
    h = (stop-start)/(n-1)
    for i=1:n
        ts[i] = start+(i-1)*h
    end
end

function linspace!(ts::Vector, start::Real, stop::Real, n1::Int, n::Int)
    h = (stop-start)/(n-1)
    for i=1:n
        ts[i+n1] = start+(i-1)*h
    end
end



#=
 lhfs
 Description:
    Computes value and possibly source-derivatives of fundamental solution at
    a series of target points and energies, with the source at the origin.
 Parameters:
    x       (input) - array of x coordinates in the cartesian plane
    y       (input) - array of y coordinates in the cartesian plane
    energies(input) - parameter array to fundamental solution, one for each target pt
    derivs  (input) - 1: compute values and derivatives to fundamental solution.
                      0: only compute the fundamental solution values.
    n       (input) - size of x, y arrays
    u       (output) - array of size n of fundamental solution values.
            u[i] is the fundamental solution at the coordinate (x[i],y[i]).
    ux      (output) - array of the x derivative of each fundamental solution value
            (unused if derivs=0)
    uy      (output) - array of the y derivative of each fundamental solution value
            (unused if derivs=0)
    stdquad (input, int)     - global convergence params: # default quad pts per saddle pt
     h (input, double)        - PTR spacing relative to std Gaussian sigma
                           (if h<0, then -h is used and h is also used to
			   scale the parameters maxh and minsaddlen)
     meth (input) - int giving method for choosing # nodes (passed to quad_nodes)
     gamout (input) - 0: no file output (default), 1: diagnostic text to file nodes.dat
=#
function lhfs(x::Float64, y::Float64, energies::Float64, derivs::Bool, stdquad::Int, h::Float64, meth::Int)
    a = 0.25(x^2+y^2)
    b = 0.5y+energies
    if a == 0
        if derivs
            u = Inf+ZIM
            ux = Inf+ZIM
            uy = Inf+ZIM
            return u,ux,uy
        else
            u = Inf+ZIM
            return u
        end
    else
        nquad = quad_nodes!(ts, ws, a, b, x, y, derivs, stdquad, h, meth)
        contour!(gam, gamp, a, b, ts, nquad)
        integrand!(integ, integx, integy, gam, a, b, x, y, derivs, nquad)
        if derivs
            u,ux,uy = quadsum3(integ,integx,integy,gamp,ws,nquad)
            return u,ux,uy
        else
            u = quadsum(integ,gamp,ws,nquad)
            return u
        end
    end
end

function lhfs!(u::Vector{Complex{Float64}}, x::Vector{Float64}, y::Vector{Float64}, energies::Vector{Float64}, derivs::Bool, stdquad::Int, h::Float64, meth::Int, n::Int)
    for i=1:n
        a = 0.25(x[i]^2+y[i]^2)
        b = 0.5y[i]+energies[i]
        if a == 0
            u[i] = Inf+ZIM
        else
            nquad = quad_nodes!(ts, ws, a, b, x[i], y[i], derivs, stdquad, h, meth)
            contour!(gam, gamp, a, b, ts, nquad)
            integrand!(integ, integx, integy, gam, a, b, x[i], y[i], derivs, nquad)
            quadsum!(u,i,integ,gamp,ws,nquad)
        end
    end
end

function lhfs!(u::Vector{Complex{Float64}}, ux::Vector{Complex{Float64}}, uy::Vector{Complex{Float64}}, x::Vector{Float64}, y::Vector{Float64}, energies::Vector{Float64}, derivs::Bool, stdquad::Int, h::Float64, meth::Int, n::Int)
    for i=1:n
        a = 0.25(x[i]^2+y[i]^2)
        b = 0.5y[i]+energies[i]
        if a == 0
            u[i] = Inf+ZIM
            ux[i] = Inf+ZIM
            uy[i] = Inf+ZIM
        else
            nquad = quad_nodes!(ts, ws, a, b, x[i], y[i], derivs, stdquad, h, meth)
            contour!(gam, gamp, a, b, ts, nquad)
            integrand!(integ, integx, integy, gam, a, b, x[i], y[i], derivs, nquad)
            quadsum3!(u,ux,uy,i,integ,integx,integy,gamp,ws,nquad)
        end
    end
end

function quadsum!(u::Vector, i::Int, integ::Vector, gamp::Vector, ws::Vector, n::Int)
    @inbounds u[i] = integ[1]*gamp[1]*ws[1]
    @simd for k=2:n
        @inbounds u[i] += integ[k]*gamp[k]*ws[k]
    end
end

function quadsum3!(u::Vector, ux::Vector, uy::Vector, i::Int, integ::Vector, integx::Vector, integy::Vector, gamp::Vector, ws::Vector, n::Int)
    @inbounds temp = gamp[1]*ws[1]
    @inbounds u[i] = integ[1]*temp
    @inbounds ux[i] = integx[1]*temp
    @inbounds uy[i] = integy[1]*temp
    @simd for k=2:n
        @inbounds temp = gamp[k]*ws[k]
        @inbounds u[i] += integ[k]*temp
        @inbounds ux[i] += integx[k]*temp
        @inbounds uy[i] += integy[k]*temp
    end
end

function quadsum(integ::Vector, gamp::Vector, ws::Vector, n::Int)
    @inbounds ret = integ[1]*gamp[1]*ws[1]
    @simd for k=2:n
        @inbounds ret += integ[k]*gamp[k]*ws[k]
    end
    ret
end

function quadsum3(integ::Vector, integx::Vector, integy::Vector, gamp::Vector, ws::Vector, n::Int)
    @inbounds temp = gamp[1]*ws[1]
    @inbounds ret1 = integ[1]*temp
    @inbounds ret2 = integx[1]*temp
    @inbounds ret3 = integy[1]*temp
    @simd for k=2:n
        @inbounds temp = gamp[k]*ws[k]
        @inbounds ret1 += integ[k]*temp
        @inbounds ret2 += integx[k]*temp
        @inbounds ret3 += integy[k]*temp
    end
    ret1,ret2,ret3
end



function lhelmfs(trg::Union{Float64,Complex{Float64}},energies::Float64;derivs::Bool=false)
    stdquad = 400
    h = 0.25
    meth = 1
    x1,x2 = reim(trg)
    lhfs(x1,x2,energies,derivs,stdquad,h,meth)
end

function lhelmfs(trg::Union{Vector{Float64},Vector{Complex{Float64}}},energies::Vector{Float64};derivs::Bool=false)
    n = length(trg)
    @assert n == length(energies)
    stdquad = 400
    h = 0.25
    meth = 1
    x1,x2 = reim(trg)
    u = Vector{Complex{Float64}}(undef,n)
    if derivs
        ux = Vector{Complex{Float64}}(undef,n)
        uy = Vector{Complex{Float64}}(undef,n)
        lhfs!(u,ux,uy,x1,x2,energies,derivs,stdquad,h,meth,n)
        return u,ux,uy
    else
        lhfs!(u,x1,x2,energies,derivs,stdquad,h,meth,n)
        return u
    end
end

function lhelmfs(trg::Union{Matrix{Float64},Matrix{Complex{Float64}}},E::Matrix{Float64};derivs::Bool=false)
    sizetrg,sizeE = size(trg),size(E)
    @assert sizetrg == sizeE

    if derivs
        u,ux,uy = lhelmfs(vec(trg),vec(E);derivs=derivs)
        return reshape(u,sizetrg),reshape(ux,sizetrg),reshape(uy,sizetrg)
    else
        u = lhelmfs(vec(trg),vec(E);derivs=derivs)
        return reshape(u,sizetrg)
    end
end

lhelmfs(trg::Union{VecOrMat{Float64},VecOrMat{Complex{Float64}}},E::Float64;derivs::Bool=false) =
    lhelmfs(trg,fill(E,size(trg));derivs=derivs)

lhelmfs(trg::Union{T1,VecOrMat{T1}},src::Union{T2,VecOrMat{T2}},E::Float64;derivs::Bool=false) where {T1<:Union{Float64,Complex{Float64}},T2<:Union{Float64,Complex{Float64}}} =
    lhelmfs(trg.-src,E.+imag.(src);derivs=derivs)
