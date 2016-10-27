require 'image'

function create_display(opt)

    function display(input, target, output)
        local batch_size = input:size()[1]

        local height = input:size()[input:dim() - 1]
        local width = input:size()[input:dim()]

        local ims = {}
        for batch_idx=1, batch_size do
            table.insert(ims, input[batch_idx]:float())

            target_rgb = torch.Tensor(3, height, width)
            target_rgb[1] = target[batch_idx]:float()
            target_rgb[2] = target[batch_idx]:float()
            target_rgb[3] = target[batch_idx]:float()
            table.insert(ims, target_rgb)

            output_rgb = torch.Tensor(3, height, width)
            output_rgb[1] = output[batch_idx]:float()
            output_rgb[2] = output[batch_idx]:float()
            output_rgb[3] = output[batch_idx]:float()
            table.insert(ims, output_rgb)
        end

        return image.toDisplayTensor{input=ims, nrow=3, padding=10, scaleeach=true}
    end

    return display
end

return create_display
