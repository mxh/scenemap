local M = {}
local SemanticDataset = torch.class('scenemap.SemanticDataset', M)

function get_data_from_daemon(start_daemon_file, n_frames_file, done_daemon_file, buffer_size, n_classes, width, height, storage_dir)
    require('paths')
    require('util/fastwrite')

    -- write number of scenes to get from daemon to file
    file = io.open(n_frames_file, "w")
    io.output(file)
    io.write(buffer_size)
    io.close(file)
    
    -- write "ready" state for daemon to file
    file = io.open(start_daemon_file, "w")
    io.output(file)
    io.write("1")
    io.close(file)
    
    -- switch back to standard output
    io.output(io.stdout)

    local data = torch.Tensor(buffer_size, width, height, 3):byte() -- this is the order in which opencv stores...
    local tmp_target = torch.Tensor(buffer_size, n_classes + 2, width, height):byte()
    local target = torch.Tensor(buffer_size, width, height):float()
    
    -- check if daemon is done
    while true do
        check_file = io.open(done_daemon_file, "r")
        if check_file then
            io.input(check_file)
            line = io.read()
            if line == "1" then
                io.close(check_file)
                check_file = io.open(done_daemon_file, "w")
                io.output(check_file)
                io.write("0")
                if check_file then io.close(check_file) end
                io.output(io.stdout)
                break
            else
                io.close(check_file)
            end
        end
    end

    fastReadByte(storage_dir .. "/rgb.bin", data)
    data = data:transpose(2, 4):transpose(3, 4):float():div(255)
    fastReadByte(storage_dir .. "/sem.bin", tmp_target)

    -- collect target maps into single map
    for i=1,buffer_size do
        mm, target[i] = torch.max(tmp_target[i], 1)
    end

    collectgarbage()

    return {data, target}
end

function SemanticDataset:__init(opt, split)
    self.split       = split
    self.batch_size  = opt.batch_size
    self.n_classes   = opt.n_classes
    self.buffer_size = opt.buffer_size
    self.epoch_size  = opt.epoch_size

    self.daemon_dir  = opt.daemon_dir
    self.storage_dir = opt.storage_dir

    self.width       = 224
    self.height      = 224

    self.thread_pool = require('util/threadpool')(2)

    self.back_buffer = {}
    self:fill_back_buffer()
    self.thread_pool:synchronize()

    self.front_buffer   = self.back_buffer
    self.buffer_pointer = 1
end

function SemanticDataset:size()
    if self.split == 'train' then
        return math.ceil(self.epoch_size / self.batch_size)
    else
        return math.ceil(5000 / self.batch_size)
    end
end

function SemanticDataset:fill_back_buffer()
    self.thread_pool:addjob(
        function(sdf, nff, ddf, buffer_size, nclasses, width, height, std, func)
            local buffer = func(sdf, nff, ddf, buffer_size, nclasses, width, height, std)
            return buffer
        end,
        function(buffer)
            self.back_buffer = buffer
        end,
        paths.concat(self.daemon_dir, 'startDaemon.txt'),
        paths.concat(self.daemon_dir, 'nFrames.txt'),
        paths.concat(self.daemon_dir, 'doneDaemon.txt'),
        self.buffer_size,
        self.n_classes,
        self.width,
        self.height,
        self.storage_dir,
        get_data_from_daemon
    )
end

function SemanticDataset:get_sample()
    if ((self.buffer_size - self.buffer_pointer) + 1) < self.batch_size then
        self.thread_pool:synchronize()
        self.front_buffer = self.back_buffer
        self.buffer_pointer = 1
        self:fill_back_buffer()
    end
    self.buffer_pointer = self.buffer_pointer + self.batch_size

    local sample = {}
    sample.input = self.front_buffer[1]:narrow(1, self.buffer_pointer - self.batch_size, self.batch_size):cuda()
    sample.target = self.front_buffer[2]:narrow(1, self.buffer_pointer - self.batch_size, self.batch_size):cuda()

    return sample
end

return M.SemanticDataset
