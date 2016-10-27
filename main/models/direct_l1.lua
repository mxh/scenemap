local nn = require 'nn'
require 'cunn'
require 'nngraph'

local Convolution = cudnn.SpatialConvolution
local ReLU = cudnn.ReLU
local Sigmoid = cudnn.Sigmoid
local Max = nn.SpatialMaxPooling
local SBatchNorm = cudnn.SpatialBatchNormalization
local BatchNorm = nn.BatchNormalization
local View = nn.View
local Linear = nn.Linear

-----------------
-- MODEL SETUP --
-----------------
function create_model(opt)
    local function layer(n, pool)
        local n_input_plane = i_channels
        i_channels = n

        local s = nn.Sequential()
        s:add(Convolution(n_input_plane, n, 3, 3, 1, 1, 1, 1))
        s:add(SBatchNorm(n, 1e-3))
        s:add(ReLU(true))
        if pool then
            s:add(Max(2, 2))
        end

        return s
    end

    local function fc_layer(n)
        local n_input_plane = i_channels
        i_channels = n

        local s = nn.Sequential()
        s:add(Linear(n_input_plane, n))
        s:add(BatchNorm(n, 1e-3))

        return s
    end

    model = nn.Sequential()

    encoder = nn.Sequential()
    i_channels = 3
    encoder:add(layer(64, true))
    encoder:add(layer(128, true))
    encoder:add(layer(256, false))
    encoder:add(layer(256, true))
    encoder:add(layer(512, false))
    encoder:add(layer(512, true))
    encoder:add(layer(512, true))

    model:add(encoder)

    model:add(Convolution(512, 1024, 7, 7, 1, 1, 0, 0))
    model:add(SBatchNorm(1024, 1e-3))
    model:add(View(1024))

    local n_out_pixels = opt.n_classes * opt.grid_size * opt.grid_size
    i_channels = 1024
    model:add(fc_layer(n_out_pixels))

    model:add(nn.Reshape(opt.n_classes, opt.grid_size, opt.grid_size))
    model:add(Sigmoid())
    model:add(nn.L1Penalty(1e-8))

    ----------------
    -- MODEL INIT --
    ----------------
    model = require('util/weight-init')(model, 'xavier')

    model = model:cuda()

    return model
end

return create_model
