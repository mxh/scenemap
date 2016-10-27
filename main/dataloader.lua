local datasets = require 'datasets/init'

local M = {}
local DataLoader = torch.class('scenemap.DataLoader', M)

function DataLoader.create(opt)
    local loaders = {}

    for i, split in ipairs{'train', 'val'} do
        local dataset = datasets.create(opt, split)
        loaders[i] = M.DataLoader(dataset, opt, split)
    end

    return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
    self.__size = dataset:size()
    self.dataset = dataset
end

function DataLoader:size()
    return self.__size
end

function DataLoader:run()
    local n = 0
    local function loop()
        if n > self.__size then
            return nil
        end
        n = n + 1
        return n, self.dataset:get_sample()
    end

    return loop
end

return M.DataLoader
