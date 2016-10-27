require 'util/fastwrite'
require 'paths'
require 'nn'

local M = {}
local FixedMapDataset = torch.class('scenemap.FixedMapDataset', M)

function pool_target(target, times)
    local pooler = nn.SpatialMaxPooling(2, 2)

    local out_target = torch.Tensor(target:size()[1], target:size()[2], 16 / (2 ^ times), 16 / (2 ^ times))

    for i=1, target:size()[1] do
        local tmp_tensor = target[i]:clone():float()
        for k=1,times do
            tmp_tensor = pooler:forward(tmp_tensor):clone()
        end
        out_target[i] = tmp_tensor
    end

    return out_target:contiguous()
end

function process_target(target, grid_size)
    local times
    if grid_size == 2 then
        times = 3
    elseif grid_size == 4 then
        times = 2
    elseif grid_size == 8 then
        times = 1
    else
        times = 0
    end

    return pool_target(target:transpose(1, 2), times):transpose(1, 2)
end

function FixedMapDataset:__init(opt, split)
    self.split       = split
    self.batch_size  = opt.batch_size
    self.n_classes   = opt.n_classes
    self.buffer_size = opt.buffer_size
    self.epoch_size  = opt.epoch_size
    self.storage_dir = opt.storage_dir

    self.grid_size   = opt.grid_size

    self.width       = 224
    self.height      = 224

    self.buffer = {}
    self.buffer.data = torch.Tensor(self.buffer_size, self.width, self.height, 3):byte()
    self.buffer.target = torch.Tensor(self.buffer_size, self.n_classes + 2, 16, 16):int()

    fastReadByte(self.storage_dir .. "/rgb.bin", self.buffer.data)
    self.buffer.data = self.buffer.data:transpose(2, 4):transpose(3, 4):float():div(255)
    fastReadInt(self.storage_dir .. "/map.bin", self.buffer.target)
    self.buffer.target = self.buffer.target:narrow(2, 3, self.n_classes):float()

    collectgarbage()

    self.buffer_pointer = 1
    self.perm = torch.randperm(self.buffer_size)
end

function FixedMapDataset:size()
    return math.ceil(self.buffer_size / self.batch_size)
end

function FixedMapDataset:get_sample()
    if ((self.buffer_size - self.buffer_pointer) + 1) < self.batch_size then
        self.buffer_pointer = 1
        self.perm = torch.randperm(self.buffer_size)
    end
    self.buffer_pointer = self.buffer_pointer + self.batch_size

    local sample = {}
    sample.input = self.buffer.data:narrow(1, self.buffer_pointer - self.batch_size, self.batch_size):cuda()
    sample.target = process_target(self.buffer.target:narrow(1, self.buffer_pointer - self.batch_size, self.batch_size), self.grid_size)

    return sample
end

return M.FixedMapDataset
