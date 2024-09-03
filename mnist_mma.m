(* https://reference.wolfram.com/language/ref/NetModel.html  *)

net = NetModel["LeNet Trained on MNIST Data"]

net[{\!\(\*
GraphicsBox[
TagBox[RasterBox[CompressedData["
1:eJxTTMoPSmNiYGAo5gASQYnljkVFiZXBAkBOaF5xZnpeaopnXklqemqRRRJI
mQwU/6cK2MHQhEtqiTTTJVxyQUy633BIneZSeolLWwSTBS6pP1ZM+bjkzjEJ
4ZL6X81kjkvqKydTHS65L0y45c4wMS3FJdfOxHRtcW3tkY/Y5cBA+T5WOaPk
5GgmJkNMOQWwW/6FMMliyqWB5X46MlliNTPo//+JTExbMOU+Ad0h683ElP8P
U+73fCGQM4XuYPHD//97LZmYLPdhlRrUAABgHMjK
"], {{0, 28}, {28, 0}}, {0, 
       255},
ColorFunction->GrayLevel],
BoxForm`ImageTag[
      "Byte", ColorSpace -> Automatic, Interleaving -> None],
Selectable->False],
DefaultBaseStyle->"ImageGraphics",
ImageSizeRaw->{28, 28},
PlotRange->{{0, 28}, {0, 28}}]\), \!\(\*
GraphicsBox[
TagBox[RasterBox[CompressedData["
1:eJxTTMoPSmNiYGAo5gASQYnljkVFiZXBAkBOaF5xZnpeaopnXklqemqRRRJI
mQwU/x9IsJWBKew2dql3okzMzH6/sMr5M7tPNGWeglXOQPf//yPiZdjlgMI7
mbHKVTOVQQkM8FCaef+v3dIih7HIfQphnnKCmdntBzYzdzKLizEzM+/DJvdD
hxnoP+xueafNxCvJwWSG3S0SV/+7M/tikbvAzHzyP1AOi5lvNJj9fwPlVJdj
ytUzMx8HUu7xWIw0ZdJ49f/PJUWTL5hyTMwT//+/xMxcgkUfE/Ou/92KzKGf
sMqpGLBzTb6MRer/Sh1gmGDzG/0AACEauS8=
"], {{0, 28}, {28, 0}}, {0, 255},
       
ColorFunction->GrayLevel],
BoxForm`ImageTag[
      "Byte", ColorSpace -> Automatic, Interleaving -> None],
Selectable->False],
DefaultBaseStyle->"ImageGraphics",
ImageSizeRaw->{28, 28},
PlotRange->{{0, 28}, {0, 28}}]\), \!\(\*
GraphicsBox[
TagBox[RasterBox[CompressedData["
1:eJxTTMoPSmNiYGAo5gASQYnljkVFiZXBAkBOaF5xZnpeaopnXklqemqRRRJI
mQwU/x9IsN/BdW1Vstx8LFK7eZgYmYBAcepvNJkvG/mZIHJMTLfR5OaBBAtX
r44CKqn4hSK1TRgoZQNipQEZp5ClPpoCRSr+gJiXgarykeWWAqUqoSalocrd
52ViCnwJMx5VLpeJyQVufyOK3Fo2JqYWOM+WiWkFkjZGRjs4p4mRkXE5nPcp
GegQGOeXM9BZ1vth3EAgT3I+lHMOyLE7Adf3y4uJqeY1hP3MFShXhbDuTwgT
0ywIa5s9Wnh+AnIFErYBgSVIRrz4G0LumxoTFIDiQaztPzLIQ5JzPogi9f+Z
NkzOof77fzTwY1EcSKps5y90GToCAMITbxU=
"], {{0, 28}, {28, 0}}, {0, 255},
       
ColorFunction->GrayLevel],
BoxForm`ImageTag[
      "Byte", ColorSpace -> Automatic, Interleaving -> None],
Selectable->False],
DefaultBaseStyle->"ImageGraphics",
ImageSizeRaw->{28, 28},
PlotRange->{{0, 28}, {0, 28}}]\)}]


net[\!\(\*
GraphicsBox[
TagBox[RasterBox[CompressedData["
1:eJxTTMoPSmNiYGAo5gASQYnljkVFiZXBAkBOaF5xZnpeaopnXklqemqRRRJI
mQwU/x9IsJWBKew2dql3okzMzH6/sMr5M7tPNGWeglXOQPf//yPiZdjlgMI7
mbHKVTOVQQkM8FCaef+v3dIih7HIfQphnnKCmdntBzYzdzKLizEzM+/DJvdD
hxnoP+xueafNxCvJwWSG3S0SV/+7M/tikbvAzHzyP1AOi5lvNJj9fwPlVJdj
ytUzMx8HUu7xWIw0ZdJ49f/PJUWTL5hyTMwT//+/xMxcgkUfE/Ou/92KzKGf
sMqpGLBzTb6MRer/Sh1gmGDzG/0AACEauS8=
"], {{0, 28}, {28, 0}}, {0, 255},
      
ColorFunction->GrayLevel],
BoxForm`ImageTag[
     "Byte", ColorSpace -> Automatic, Interleaving -> None],
Selectable->False],
DefaultBaseStyle->"ImageGraphics",
ImageSizeRaw->{28, 28},
PlotRange->{{0, 28}, {0, 28}}]\), "Probabilities"]


net[\!\(\*
GraphicsBox[
TagBox[RasterBox[CompressedData["
1:eJxTTMoPSmNiYGAo5gASQYnljkVFiZXBAkBOaF5xZnpeaopnXklqemqRRRJI
mQwU/x9IsJWBKew2dql3okzMzH6/sMr5M7tPNGWeglXOQPf//yPiZdjlgMI7
mbHKVTOVQQkM8FCaef+v3dIih7HIfQphnnKCmdntBzYzdzKLizEzM+/DJvdD
hxnoP+xueafNxCvJwWSG3S0SV/+7M/tikbvAzHzyP1AOi5lvNJj9fwPlVJdj
ytUzMx8HUu7xWIw0ZdJ49f/PJUWTL5hyTMwT//+/xMxcgkUfE/Ou/92KzKGf
sMqpGLBzTb6MRer/Sh1gmGDzG/0AACEauS8=
"], {{0, 28}, {28, 0}}, {0, 255},
      
ColorFunction->GrayLevel],
BoxForm`ImageTag[
     "Byte", ColorSpace -> Automatic, Interleaving -> None],
Selectable->False],
DefaultBaseStyle->"ImageGraphics",
ImageSizeRaw->{28, 28},
PlotRange->{{0, 28}, {0, 28}}]\), "Probabilities"]

