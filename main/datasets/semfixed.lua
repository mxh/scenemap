require 'util/fastwrite'
require 'paths'

local M = {}
local FixedSemanticDataset = torch.class('scenemap.FixedSemanticDataset', M)

function FixedSemanticDataset:__init(opt, split)
    self.split       = split
    self.batch_size  = opt.batch_size
    self.n_classes   = opt.n_classes
    self.buffer_size = opt.buffer_size
    self.epoch_size  = opt.epoch_size
    self.storage_dir = opt.storage_dir

    self.width       = 224
    self.height      = 224

    self.buffer = {}
    self.buffer.data = torch.Tensor(self.buffer_size, self.width, self.height, 3):byte()
    self.buffer.target = torch.Tensor(self.buffer_size, self.width, self.height):float()

    fastReadByte(self.storage_dir .. "/rgb.bin", self.buffer.data)
    self.buffer.data = self.buffer.data:transpose(2, 4):transpose(3, 4):float():div(255)
    local tmp_target = torch.Tensor(self.buffer_size, self.n_classes + 2, self.width, self.height):byte()
    fastReadByte(self.storage_dir .. "/sem.bin", tmp_target)
    for i=1, self.buffer_size do
        mm, self.buffer.target[i] = torch.max(tmp_target[i], 1)
    end

    collectgarbage()

    self.buffer_pointer = 1
end

function FixedSemanticDataset:size()
    return math.ceil(self.buffer_size / self.batch_size)
end

function FixedSemanticDataset:get_sample()
    if ((self.buffer_size - self.buffer_pointer) + 1) < self.batch_size then
        self.buffer_pointer = 1
    end
    self.buffer_pointer = self.buffer_pointer + self.batch_size

    local sample = {}
    sample.input = self.buffer.data:narrow(1, self.buffer_pointer - self.batch_size, self.batch_size):cuda()
    sample.target = self.buffer.target:narrow(1, self.buffer_pointer - self.batch_size, self.batch_size):cuda()

    return sample
end

return M.FixedSemanticDataset
