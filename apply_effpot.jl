include("band_diagrams_bm_like.jl")

#################### Second step : compute the effective potentials 𝕍, 𝕎, 𝔸, etc. Very rapid

# Computes effective potentials, makes plots, gives some information
function computes_and_plots_effective_potentials()
    # Basic parameters
    N = 27; Nz = 600; gauge_param = 1

    # Set parameters and makes imports of eigenfunctions
    compute_Vint = true # include Vint
    interlayer_distance = 6.45
    p = import_and_computes(N,Nz,gauge_param,compute_Vint,interlayer_distance) # imports the eigenfunctions Φ1 and Φ2
    p.plots_cutoff = 5
    update_plots_resolution(200,p)
    p.plots_n_motifs = 6
    produce_secondary_plots = false
    creates_δ𝕍(p) # 𝕍 - T_BM
    p.plot_for_article = true
    px("Computes and plots, for N ",N,", Nz ",Nz,", gauge_param ",gauge_param," d ",p.interlayer_distance)

    # Comparisions between the Bistritzer-MacDonald potential, and 𝕍_V (or Σ)
    compare_to_BM_infos(p.𝕍_V,p,"V_V") # it is normal that individual blocks distances are half the total distance because there are two blocks each time
    compare_to_BM_infos(p.Σ,p,"Σ")

    # Prints some info
    # get_low_Fourier_coefs(p) # print low Fourier modes of the effective potentials
    W_without_mean = print_infos_W(p) # prints some information about W

    ####################### Tests symmetries

    info_particle_hole_symmetry(p)
    info_PT_symmetry(p)
    info_translation_symmetry(p)
    info_mirror_symmetry(p)
    test_sym_Wplus_Wminus(p) # special symmetries of W
    # info_equality_some_blocks_symmetry(p)
    px("Compare W+ and W- ",distance(p.W_V_plus,p.W_V_minus))
    px("Distance 𝕍_{11} - 𝕍_{22} ",relative_distance_blocks(p.𝕍[1],p.𝕍[4]))
    test_block_hermitianity(p.Wplus_tot,p;name="W")

    ####################### Plots
    
    # Plots for article
    if p.plot_for_article
        # plot_block_article(p.δ𝕍,p;title="δV",k_red_shift=-p.m_q1)
        # plot_block_article(p.T_BM,p;title="T",k_red_shift=-p.m_q1)
        # plot_block_article(W_without_mean,p;title="W_plus_without_mean")
        # plot_block_article(p.𝕍_V,p;title="V_V")
        # plot_block_article(p.𝔸1,p;title="A",other_block=p.𝔸2,k_red_shift=-p.m_q1,meV=false,coef=1/p.vF,vertical_bar=true)
        # if p.compute_Vint plot_block_article(p.𝕍_Vint,p;title="V_Vint",k_red_shift=-p.m_q1) end
        # plot_block_article(p.Σ,p;title="Σ",k_red_shift=-p.m_q1,meV=false)
        # plot_block_article(p.W_non_local_plus,p;title="W_nl_plus",k_red_shift=-p.m_q1,vertical_bar=true)
    end

    if produce_secondary_plots
        # Plots in reduced coordinates
        plot_block_reduced(p.T_BM,p;title="T")
        plot_block_reduced(p.Wplus_tot,p;title="W")
        plot_block_reduced(p.𝕍,p;title="V")
        plot_block_reduced(p.Σ,p;title="Σ")
        plot_block_reduced(p.𝔸1,p;title="A1")
        plot_block_reduced(p.𝔸2,p;title="A2")

        # Plots in cartesian coordinates
        plot_block_cart(p.T_BM,p;title="T",article=true)

        # W
        plot_block_cart(p.W_V_plus,p;title="W_V_plus",article=true)
        plot_block_cart(p.W_V_minus,p;title="W_V_minus")
        plot_block_cart(p.W_non_local_plus,p;title="W_nl_plus",article=true)
        plot_block_cart(p.W_non_local_minus,p;title="W_nl_moins")
        plot_block_cart(p.Wplus_tot,p;title="W_plus_tot")

        # V
        plot_block_cart(p.𝕍,p;title="V")
        plot_block_cart(p.𝕍_V,p;title="V_V",article=true)
        plot_block_cart(p.𝕍_Vint,p;title="V_Vint",article=true)

        # Σ and A
        plot_block_cart(p.Σ,p;title="Σ",article=true)
        plot_magnetic_block_cart(p.𝔸1,p.𝔸2,p;title="A",article=true)
    end
    p
    nothing
end

computes_and_plots_effective_potentials()
