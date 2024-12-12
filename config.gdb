target ext :3333
load

# interface with asm, regs and cmd windows
define sasm
  layout split
  layout asm
  layout regs
  focus cmd
end

# interface with C source, regs and cmd windows
define sc
  layout split
  layout src
  layout regs
  focus cmd
end