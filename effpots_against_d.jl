include("band_diagrams_bm_like.jl")
using DelimitedFiles, CairoMakie, LaTeXStrings

# Creates the graph
# "fine" is for the log figures (left and middle)
# "graphs" is the the left one, "!graphs" for the middle one
function save_fig(list_d,meas,measures,id_graph,labels,colors,p,fine,graphs_V)
    res = 500
    resX_not_fine = 200
    resX_fine = 350
    meV = fine
    c_meV = meV ? 1e3 : 1
    fig = CairoMakie.Figure(resolution=(fine ? resX_fine : resX_not_fine,res))

    ax = CairoMakie.Axis(fig[1, 1], xlabel = "d (Bohr)")#, ylabel = "∅")
    xlim = fine ? maximum(list_d) : 7
    ax.xticks = (0:1:xlim)
    CairoMakie.xlims!(ax,0,xlim)

    # Limits, logscale, annotation
    if fine
        minY2 = 1e-5
        maxY2 = 1e2
        X = 6.45
        CairoMakie.vlines!(ax,[X],color = :black)
        CairoMakie.text!(string("d=",X), position=(X+0.2,minY2*1.1), textsize =15)
        CairoMakie.ylims!(ax,minY2,maxY2)
    end
    lin = []
    fc = fine ? abs : x-> x

    # Includes lines
    for i=1:length(meas)
        mea = meas[i]
        if !fine || (fine && graphs_V && id_graph[i]=="1") || (fine && !graphs_V && id_graph[i]=="2")
            l = CairoMakie.lines!(ax, list_d, fc.(measures[mea]),label=labels[i],color=colors[i],linestyle=nothing)
            push!(lin,l)
        end
    end
    if fine
        ax.yscale = log10
    end

    # Legend
    # figlegend = CairoMakie.Figure(resolution=(1000,300))
    # if !fine
    # patchsize = 10
    # CairoMakie.Legend(figlegend[1, 2],lin,labels,framevisible = true,patchsize = (patchsize, patchsize),nbanks=5)#,fontsize=20)
    # axislegend(ax; labelsize=5)
    # end
    
    # Saves
    add_name = !fine ? "" : (graphs_V ? "_log_V" : "_log_others")
    post_path = string("study_d",add_name,".pdf")
    for pre in ["effective_potentials/",p.article_path]
        CairoMakie.save(string(pre,post_path),fig)
        px("SAVES")
    end
end

# Produces the study of effective potentials magnitudes against d (the interlayer distance) corresponding to the article "A simple derivation of moiré-scale continuous models for twisted bilayer graphene"
function study_in_d() # curves with and without Vint are extremely close
    # N = 24; Nz = 432; gauge_param = 1
    # N = 24; Nz = 125; gauge_param = 1
    N = 27; Nz = 600; gauge_param = 1 # <--- ecut 40|Kd|^2, L = 125
    px("N ",N,", Nz ",Nz)
    compute_Vint = true # include Vint
    list_d = sort(vcat([6.45],[0.01],(0.1:0.1:11))) # set of values of d of the study

    ######### Prepares the measures we are going to make
    measures = Dict()
    meas = ["wAA","wC","wΣ","norm_∇Σ","norm_W_without_mean","distV","distΣ"] #"Wmean"
    id_graph = ["1","1","2","2","2","1","2"]
    wAA_str = "w_{AA}^{d=6.45}"
    cf = 4
    ps(s) = string("\\frac{1}{",cf,"} {\\Vert}",s,"\\Vert_{L^2}")
    function cS(a,b=wAA_str;norm=false)
        aa = norm ? ps(a) : a
        LaTeXString(string("\$\\frac{",aa,"}{",b,"}\$")) # s/wAA
    end
    labels = [cS("w_{AA}"),cS("[V_{d} u_1 \\mid u_1]_{d,-1,-1}"),L"$[u_1,u_1]_{d,0,0}$",cS("∇\\Sigma_d","v_F";norm=true),cS("𝕎_d - W_d 𝕀 ";norm=true),cS("𝕍_d - [V u_1,u_1]_{d,0,0} 𝐕 - [V u_1,u_1]_{d,-1,-1} 𝐕";norm=true),LaTeXString(string("\$",ps("\\Sigma_d - [u_1,u_1]_{d,0,0} 𝐕"),"\$"))] # cS("W_d")
    colors = [:blue,:green1,:pink,:red,:black,:brown,:orange,:cyan,:darkgreen]
    for m in meas
        measures[m] = zeros(Float64,length(list_d))
    end

    ######### Computes wAA for d=6.45, which is the energy reference
    interlayer_distance = 6.45
    p = import_and_computes(N,Nz,gauge_param,compute_Vint,interlayer_distance) # imports the eigenfunctions Φ1 and Φ2
    c = 1/sqrt(p.cell_area)
    wAA_ref = p.wAA/c

    ######### Computes quantities for each d
    for i=1:length(list_d)
        d = list_d[i]
        p = import_and_computes(N,Nz,gauge_param,compute_Vint,d)
        V = p.compute_Vint ? p.𝕍 : p.𝕍_V
        W = p.compute_Vint ? p.Wplus_tot : p.W_V_plus

        # V and Σ
        wAA = real(V[1][1,1])
        wC = real(V[1][end,2])
        wD = real(V[1][1,2])
        wΣ = real(p.Σ[1][1,1])
        sm = op_two_blocks((x,y)->x.+y,T_BM_four(wAA,wAA,p),T_BM_four(wC,wC,p;second=true))

        # W
        mean_W_block = mean_block(W,p)
        W_without_mean = add_cst_block(W,-mean_W_block/c,p)
        # meanW = real(mean_W_block[1,1])

        measures["wAA"][i] = wAA/wAA_ref
        measures["wC"][i] = wC/wAA_ref
        measures["wΣ"][i] = c*wΣ

        measures["distV"][i] = norm_block(op_two_blocks((x,y)->x.-y,sm,V),p)/wAA_ref
        measures["distΣ"][i] = c*norm_block(op_two_blocks((x,y)->x.-y,T_BM_four(wΣ,wΣ,p),p.Σ),p)
        measures["norm_∇Σ"][i] = c*norm_block_potential(p.𝔸1,p.𝔸2,p)/p.vF
        measures["norm_W_without_mean"][i] = norm_block(W_without_mean,p)/wAA_ref
        px(p.interlayer_distance," ")
    end
    # Print measures
    for i=1:length(meas)
        mea = meas[i]
        px(mea,"\n",measures[mea],"\n")
    end
    # Create the graphs
    create_fig(fine,graphs_V) = save_fig(list_d,meas,measures,id_graph,labels,colors,p,fine,graphs_V)
    create_fig(true,true)
    create_fig(true,false)
    create_fig(false,true)
end

study_in_d() # curves with and without Vint are extremely close
