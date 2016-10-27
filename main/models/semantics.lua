local nn = require 'nn'
require 'cunn'
require 'nngraph'

local Convolution = cudnn.SpatialConvolution
local ReLU = cudnn.ReLU
local Sigmoid = cudnn.Sigmoid
local Max = nn.SpatialMaxPooling
local UnMax = nn.SpatialMaxUnpooling
local SBatchNorm = cudnn.SpatialBatchNormalization
local BatchNorm = nn.BatchNormalization

function create_model(opt)
    local function add_layer(n, pool, model, no_relu)
        local n_input_plane = i_channels
        i_channels = n

        local s = nn.Sequential()
        s:add(Convolution(n_input_plane, n, 3, 3, 1, 1, 1, 1))
        s:add(SBatchNorm(n, 1e-3))

        if not no_relu then
            s:add(ReLU(true))
        end

        local smp
        if pool then
            smp = Max(2, 2)
            s:add(smp)
        else
            smp = nil
        end

        model:add(s)

        return smp
    end

    function add_unpool(smp, model)
        model:add(UnMax(smp))
    end

    model = nn.Sequential()

    encoder = nn.Sequential()
    i_channels = 3
    smp1 = add_layer(64,  true,  encoder)
    smp2 = add_layer(128, true,  encoder)
           add_layer(256, false, encoder)
    smp3 = add_layer(256, true,  encoder)
           add_layer(512, false, encoder)
    smp4 = add_layer(512, true,  encoder)
    smp5 = add_layer(512, true,  encoder)

    model:add(encoder)

    decoder = nn.Sequential()
    add_unpool(smp5, decoder)
    add_layer(512, false, decoder)
    add_unpool(smp4, decoder)
    add_layer(512, false, decoder)
    add_layer(256, false, decoder)
    add_unpool(smp3, decoder)
    add_layer(256, false, decoder)
    add_layer(128, false, decoder)
    add_unpool(smp2, decoder)
    add_layer(64, false, decoder)
    add_unpool(smp1, decoder)

    model:add(decoder)

    add_layer(opt.n_classes + 2, false, model, true)

    ----------------
    -- MODEL INIT --
    ----------------
    model = require('util/weight-init')(model, 'xavier')

    model = model:cuda()

    return model
end

return create_model
