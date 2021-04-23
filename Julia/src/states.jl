using OffsetArrays, NCDatasets, Parameters

# Define the contents of the State Struct 
# The Parameters.jl @with_kw "macro" gives the ability to "unpack" variables when this struct is passed as an argument to functions
@with_kw struct State{T, S}
    isd        ::Int32
    ied        ::Int32
    jsd        ::Int32
    jed        ::Int32
    is         ::Int32
    ie         ::Int32
    js         ::Int32
    je         ::Int32
    nord       ::Int32
    npx        ::Int32
    npy        ::Int32
    npz        ::Int32
    dt2        ::Float64
    sw_corner  ::Int32
    se_corner  ::Int32
    ne_corner  ::Int32
    nw_corner  ::Int32
    
    ni         ::Int64
    nj         ::Int64
    nip1       ::Int64
    njp1       ::Int64
    nk         ::Int64
    nk9        ::Int64
    
    rarea      ::T
    rarea_c    ::T
    sin_sg     ::S
    cos_sg     ::S
    sina_v     ::T
    cosa_v     ::T
    sina_u     ::T
    cosa_u     ::T
    fC         ::T
    rdxc       ::T
    rdyc       ::T
    dx         ::T
    dy         ::T
    dxc        ::T
    dyc        ::T
    cosa_s     ::T
    rsin_u     ::T
    rsin_v     ::T
    rsin2      ::T
    dxa        ::T
    dya        ::T
    delpc      ::S
    delp       ::S
    ptc        ::S
    pt         ::S
    u          ::S
    v          ::S
    w          ::S
    uc         ::S
    vc         ::S
    ua         ::S
    va         ::S
    wc         ::S
    ut         ::S
    vt         ::S
    divg_d     ::S
    
end

# State function which assigns NetCDF variables,dimensions and attributes to the State struct
function State(ds::NCDataset)

    isd =                 ds.attrib["isd"]
    ied =                 ds.attrib["ied"]
    jsd =                 ds.attrib["jsd"]
    jed =                 ds.attrib["jed"]
    is =                  ds.attrib["is"]
    ie =                  ds.attrib["ie"]
    js =                  ds.attrib["js"]
    je =                  ds.attrib["je"]
    nord =                ds.attrib["nord"]
    npx =                 ds.attrib["npx"]
    npy =                 ds.attrib["npy"]
    npz =                 ds.attrib["npz"]
    dt2 =                 ds.attrib["dt2"]
    sw_corner =           ds.attrib["sw_corner"]
    se_corner =           ds.attrib["se_corner"]
    nw_corner =           ds.attrib["nw_corner"]
    ne_corner =           ds.attrib["ne_corner"]
    ni =                  ds.dim["ni"]
    nj =                  ds.dim["nj"]
    nip1 =                ds.dim["nip1"]
    njp1 =                ds.dim["njp1"]
    nk =                  ds.dim["nk"]
    nk9 =                 ds.dim["nk9"]

    
    return State(
      isd,
      ied,
      jsd,
      jed,
      is,
      ie,
      js,
      je,
      nord,
      npx,
      npy,
      npz,
      dt2,
      sw_corner,
      se_corner,
      nw_corner,
      ne_corner,
      ni,
      nj,
      nip1,
      njp1,
      nk,
      nk9,

      # Assign OffsetArrays
      # The first argument to OffsetArray is the NetCDF variable loaded in memory as a Julia array
      # The second, third (and fourth) args are the non-standardard indexed dimensions
      OffsetArray(Array(ds["rarea"]),        isd:ied,      jsd:jed     ),
      OffsetArray(Array(ds["rarea_c"]),      isd:ied + 1,  jsd:jed + 1 ),
      OffsetArray(Array(ds["sin_sg"]),       isd:ied,      jsd:jed, 1:9),
      OffsetArray(Array(ds["cos_sg"]),       isd:ied,      jsd:jed, 1:9),
      OffsetArray(Array(ds["sina_v"]),       isd:ied,      jsd:jed + 1 ),
      OffsetArray(Array(ds["cosa_v"]),       isd:ied,      jsd:jed + 1 ),
      OffsetArray(Array(ds["sina_u"]),       isd:ied + 1,  jsd:jed     ),
      OffsetArray(Array(ds["cosa_u"]),       isd:ied + 1,  jsd:jed     ),
      OffsetArray(Array(ds["fC"]),           isd:ied + 1,  jsd:jed + 1 ),
      OffsetArray(Array(ds["rdxc"]),         isd:ied + 1,  jsd:jed     ),
      OffsetArray(Array(ds["rdyc"]),         isd:ied,      jsd:jed + 1 ),
      OffsetArray(Array(ds["dx"]),           isd:ied,      jsd:jed + 1 ),
      OffsetArray(Array(ds["dy"]),           isd:ied + 1,  jsd:jed     ),
      OffsetArray(Array(ds["dxc"]),          isd:ied + 1,  jsd:jed     ),
      OffsetArray(Array(ds["dyc"]),          isd:ied,      jsd:jed + 1 ),
      OffsetArray(Array(ds["cosa_s"]),       isd:ied,      jsd:jed     ),
      OffsetArray(Array(ds["rsin_u"]),       isd:ied + 1,  jsd:jed     ),
      OffsetArray(Array(ds["rsin_v"]),       isd:ied,      jsd:jed + 1 ),
      OffsetArray(Array(ds["rsin2"]),        isd:ied,      jsd:jed     ),
      OffsetArray(Array(ds["dxa"]),          isd:ied,      jsd:jed     ),
      OffsetArray(Array(ds["dya"]),          isd:ied,      jsd:jed     ),
      OffsetArray(Array(ds["delpc"]),        isd:ied,      jsd:jed,     1:npz),
      OffsetArray(Array(ds["delp"]),         isd:ied,      jsd:jed,     1:npz),
      OffsetArray(Array(ds["ptc"]),          isd:ied,      jsd:jed,     1:npz),
      OffsetArray(Array(ds["pt"]),           isd:ied,      jsd:jed,     1:npz),
      OffsetArray(Array(ds["u"]),            isd:ied,      jsd:jed + 1, 1:npz),
      OffsetArray(Array(ds["v"]),            isd:ied + 1,  jsd:jed,     1:npz),
      OffsetArray(Array(ds["w"]),            isd:ied,      jsd:jed,     1:npz),
      OffsetArray(Array(ds["uc"]),           isd:ied + 1,  jsd:jed,     1:npz),
      OffsetArray(Array(ds["vc"]),           isd:ied,      jsd:jed + 1, 1:npz),
      OffsetArray(Array(ds["ua"]),           isd:ied,      jsd:jed,     1:npz),
      OffsetArray(Array(ds["va"]),           isd:ied,      jsd:jed,     1:npz),
      OffsetArray(Array(ds["wc"]),           isd:ied,      jsd:jed,     1:npz),
      OffsetArray(Array(ds["ut"]),           isd:ied,      jsd:jed,     1:npz),
      OffsetArray(Array(ds["vt"]),           isd:ied,      jsd:jed,     1:npz),
      OffsetArray(Array(ds["divg_d"]),       isd:ied + 1,  jsd:jed + 1, 1:npz)
    )

    # Close the .nc file 
    close(ds)

  end # function State
