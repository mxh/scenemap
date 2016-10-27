function fastWriteFloat(filename, tensor)
    f = torch.DiskFile(filename, "w"):binary()
    f:writeFloat(tensor:storage())
end

function fastReadFloat(filename, tensor)
    f = torch.DiskFile(filename, "r"):binary()
    f:readFloat(tensor:storage())
end

function fastWriteByte(filename, tensor)
    f = torch.DiskFile(filename, "w"):binary()
    f:writeByte(tensor:storage())
end

function fastReadByte(filename, tensor)
    f = torch.DiskFile(filename, "r"):binary()
    f:readByte(tensor:storage())
end

function fastReadInt(filename, tensor)
    f = torch.DiskFile(filename, "r"):binary()
    f:readInt(tensor:storage())
end
