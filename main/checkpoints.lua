local checkpoint = {}

function checkpoint.latest(opt)
    if opt.resume == 'none' then
        return nil
    end

    local latest_path = paths.concat(opt.resume, 'latest.t7')
    if not paths.filep(latest_path) then
        return nil
    end

    print('=> Loading checkpoint ' .. latest_path)
    local latest = torch.load(latest_path)
    local optim_state = torch.load(paths.concat(opt.resume, latest.optim_file))
    return latest, optim_state
end

function checkpoint.save(epoch, model, optim_state)
    if torch.type(model) == 'nn.DataParallelTable' then
        model = model:get(1)
    end

    local model_file = 'best_model.t7'
    local optim_file = 'optim_state.t7'

    torch.save(model_file, model)
    torch.save(optim_file, optim_state)
    torch.save('latest.t7', {
        epoch = epoch,
        model_file = model_file,
        optim_file = optim_file,
    })

end

return checkpoint
