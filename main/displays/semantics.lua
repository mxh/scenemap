require 'image'
require 'cudnn'
require 'util/colour_labels'

function create_display(opt)

    function display(input, target, output)
        local batch_size = input:size()[1]

        local height = input:size()[input:dim() - 1]
        local width = input:size()[input:dim()]

        local sm = cudnn.SpatialSoftMax():cuda()

        local ims = {}
        for batch_idx=1, batch_size do
            local sm_output = sm:forward(output[batch_idx]):float()
            local target_split = torch.Tensor(opt.n_classes + 2, height, width)
            for class_idx=1, opt.n_classes + 2 do
                target_split[class_idx] = torch.eq(target[batch_idx], class_idx):float()
            end
            mm, labels_target = torch.max(target_split:squeeze(), 1)
            mm, labels_output = torch.max(sm_output:squeeze(), 1)
            table.insert(ims, input[batch_idx]:float())
            table.insert(ims, colour_labels(labels_target))
            table.insert(ims, colour_labels(labels_output))
        end

        return image.toDisplayTensor{input=ims, nrow=3, padding=10}
    end

    return display
end

return create_display
