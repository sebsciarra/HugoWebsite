
from manim import *
import numpy as np


# config.frame_width = 10
# config.frame_height = 15
config.background_color = ManimColor('#002241')

def add_standard_basis(self, remove_elements_at_end=False): 
    
    # add standard basis showing to repeat definition of vector
    # region Scene 1: Vector in standard basis 
    vec_g = Vector([2, 1], color='#d170c7')
    vec_g_label = MathTex(r'\mathbf{g}', color="#d170c7", font_size=35)
    vec_g_label.align_to(vec_g, UR)
    vec_g_label.add_background_rectangle()
    vec_g_label.shift(0.5*UP)

    self.add_vector(vec_g)
    show_vec_g_label = [Write(vec_g_label)]
    self.play(*show_vec_g_label)
    self.wait(0.5)

    # endregion 

    # region Scene 2: Basis vector x
    basis_x = Vector([1, 0], color="#4EF716")
    
    # add label for basis_x 
    label_x = MathTex(r'\mathbf{b}_{\mathbf{e_x}}', color="#4EF716", font_size=35)  
    label_x.align_to(basis_x, UL)
    label_x.shift(0.25 * DOWN + 0.45 * RIGHT)

    self.add_vector(basis_x)
    self.wait(0.5)

    # endregion 

    # region Scene 3: First column with vector label x
    text = (MathTex(
            r"\begin{array}{c}\begin{matrix}" + 
            r"\hspace{1cm} \mathbf{b_{e_x}}\end{matrix} \\ " +
            r" \mathbf{B}_e = \begin{bmatrix} 1 \\ \\" + 
            r"0  \end{bmatrix} \end{array}",font_size=35). 
            to_edge(UL))
                     
    
    # add custom colouring
    # self.add(index_labels(text[0]))  # help determine what index values to colour
    text[0][0:3].set_color("#4EF716")
    text.add_background_rectangle()

    # Display the initial text
    show_text = [Create(text), Write(label_x)]
    self.play(*show_text)
    self.wait(0.5)

    # endregion 

    # region Scene 4: Basis vector y
    basis_y = Vector([0, 1], color="#FB699D")
        
    # add label for basis_y
    label_y = MathTex(r'\mathbf{b}_{\mathbf{e_y}}', color="#FB699D", font_size=35)  
    label_y.align_to(basis_y, UR)
    label_y.shift(0.3 * LEFT)

    self.add_vector(basis_y)
    self.wait(0.5)
    # endregion 

    # region Scene 5: Second column with vector label y

    text2 = (MathTex(r"\begin{array}{c}\begin{matrix}" + 
                       r"\hspace{1cm} \mathbf{b_{e_x}} & \mathbf{b_{e_y}} \end{matrix} \\ " +
                       r" \mathbf{B}_e = \begin{bmatrix} 1 & \quad 0 \\ \\" + 
                       r"0 & \quad 1 \end{bmatrix} \end{array}",font_size=35)
                       .to_edge(UL))

    # add custom colouring
    # self.add(index_labels(text[0]))  # help determine what index values to colour
    text2[0][0:3].set_color("#4EF716")
    text2[0][3:6].set_color("#FB699D")
    text2.add_background_rectangle()
    
    self.play(Transform(text, text2), Write(label_y))
    self.wait(0.5)
    # endregion

    # region Scene 6: Add x component of vector g

    vec_basis_x = Vector(color="#4EF716")
    vec_basis_x.put_start_and_end_on(start=[0,0, 0], end=[2, 0, 0])
    g_coord_x = MathTex(r'2\mathbf{b}_{\mathbf{e_x}}', color="#4EF716", font_size=35)  
    g_coord_x.add_background_rectangle()
    g_coord_x.align_to(vec_basis_x, UL)
    g_coord_x.shift(1.5 * RIGHT + 0.45 * DOWN)


    self.add_vector(vec_basis_x) 
    self.wait()
    self.play(Write(g_coord_x))

    # endregion 
      
    # region Scene 7: Add y component of vector g and move to tip of x component
    vec_basis_y = Vector(color="#FB699D")
    vec_basis_y.put_start_and_end_on(start=[0,0, 0], end=[0, 1, 0])
    g_coord_y = MathTex(r'1\mathbf{b}_{\mathbf{e_y}}', color="#FB699D", font_size=35)  
    g_coord_y.align_to(vec_basis_y, DL)
    g_coord_y.shift(0.3 * RIGHT + 0.6 * UP)

    # Create a VGroup to include both the vector and its label
    vector_with_label = VGroup(vec_basis_y, g_coord_y)

    self.add_vector(vec_basis_y) 
    self.wait()
    self.play(Write(g_coord_y))

    vec_basis_y.put_start_and_end_on(start=ORIGIN, end=[0, 1, 0])
    animation = ApplyMethod(vector_with_label.shift, [2, 0, 0])
    self.play(animation)

    # endregion 

    # region Scene 8: Add vector g
    label_vec_g = MathTex(r'\mathbf{g_e} = [2, 1]', color='#d170c7', font_size=35)
    label_vec_g.add_background_rectangle()
    label_vec_g.move_to([4.5, 0.5, 0])

    self.play(Write(label_vec_g))
    self.wait(1.5)

    obj_remove = VGroup(basis_x, label_x, basis_y, label_y, 
                        vec_basis_x, g_coord_x, vec_basis_y, g_coord_y)

    if remove_elements_at_end: 
      self.play(FadeOut(obj_remove))
    
    # endregion 


def add_non_standard_basis(self, remove_elements_at_end=False):
    

    # region Scene 1: Basis vector x
    basis_x = Vector([1, 0], color="#4EF716")
    
    # add label for basis_x 
    label_x = MathTex(r'\mathbf{b}_{\mathbf{n_x}}', color="#4EF716", font_size=35)  
    label_x.align_to(basis_x, UL)
    label_x.shift(0.25 * DOWN + 0.3 * RIGHT)

    self.add_vector(basis_x)
    self.wait(0.5)  
    # endregion 

    # region Scene 2: First column with vector label x
    text = (MathTex(
            r"\begin{array}{c}\begin{matrix}" + 
            r"\hspace{1cm} \mathbf{b_{n_x}}\end{matrix} \\ " +
            r" \mathbf{B}_n = \begin{bmatrix} 1 \\ \\" + 
            r"0  \end{bmatrix} \end{array}",font_size=35). 
            to_edge(UR))
                     
      
    # add custom colouring
    # self.add(index_labels(text[0]))  # help determine what index values to colour
    text[0][0:3].set_color("#4EF716")
    text.add_background_rectangle()

    # Display the initial text
    show_text = [Create(text), Write(label_x)]
    self.play(*show_text)
    self.wait(0.5)

    # endregion 

    # region Scene 3: Basis vector y
    basis_y = Vector([1, 2], color="#FB699D")
        
    # add label for basis_y
    label_y = MathTex(r'\mathbf{b}_{\mathbf{n_y}}', color="#FB699D", font_size=35)  
    label_y.align_to(basis_y, UR)
    label_y.add_background_rectangle()
    label_y.shift(0.3*LEFT + 0.5*UP)

    self.add_vector(basis_y)
    self.wait(0.5)
    # endregion 

    # region Scene 4: Second column with vector label y

    text2 = (MathTex(r"\begin{array}{c}\begin{matrix}" + 
                       r"\hspace{1cm} \mathbf{b_{n_x}} & \mathbf{b_{n_y}} \end{matrix} \\ " +
                       r" \mathbf{B}_n = \begin{bmatrix} 1 & \quad 1 \\ \\" + 
                       r"0 & \quad 2 \end{bmatrix} \end{array}",font_size=35)
                       .to_edge(UR))

    # add custom colouring
    # self.add(index_labels(text[0]))  # help determine what index values to colour
    text2[0][0:3].set_color("#4EF716")
    text2[0][3:6].set_color("#FB699D")
    text2.add_background_rectangle()
    
    self.play(Transform(text, text2), Write(label_y))
    self.wait(0.5)
    # endregion

    # region Scene 5: Add x component of vector g

    vec_basis_x = Vector(color="#4EF716")
    vec_basis_x.put_start_and_end_on(start=[0,0, 0], end=[1.5, 0, 0])
    g_coord_x = MathTex(r'1.5\mathbf{b}_{\mathbf{n_x}}', color="#4EF716", font_size=35)  
    g_coord_x.add_background_rectangle()
    g_coord_x.align_to(vec_basis_x, UL)
    g_coord_x.shift(1.3*RIGHT + 0.45*DOWN)


    self.add_vector(vec_basis_x) 
    self.wait()
    self.play(Write(g_coord_x))

    # endregion 
      
    # region Scene 6: Add y component of vector g and move to tip of x component

    vec_basis_y = Vector(color="#FB699D")
    vec_basis_y.put_start_and_end_on(start=[0, 0, 0], end=[0.5, 1, 0])
    g_coord_y = MathTex(r'0.5\mathbf{b}_{\mathbf{n_y}}', color="#FB699D", font_size=35)  
    g_coord_y.align_to(vec_basis_y, DL)
    g_coord_y.add_background_rectangle()
    g_coord_y.shift(0.8*LEFT + 0.5*UP)

    # Create a VGroup to include both the vector and its label
    vector_with_label = VGroup(vec_basis_y, g_coord_y)

    self.add_vector(vec_basis_y) 
    self.wait()
    self.play(Write(g_coord_y))

    vec_basis_y.put_start_and_end_on(start=ORIGIN, end=[0.5, 1, 0])
    move_basis_y = ApplyMethod(vector_with_label.shift, [1.5, 0, 0])
    move_g_coord_y = ApplyMethod(g_coord_y.shift, [3, 0.1, 0])
    animation_group = AnimationGroup(move_basis_y, move_g_coord_y)
    self.play(animation_group)

    # endregion 

    # region Scene 7: Add vector h 
    label_vec_g = MathTex(r'\mathbf{g_n} = [1.5, 0.5]', color='#d170c7', font_size=35)
    label_vec_g.add_background_rectangle()
    label_vec_g.move_to([4.75, 1, 0])

    self.play(Write(label_vec_g))
    self.wait(2)
    
    # endregion

  
class standardBasis(LinearTransformationScene):
  
  def __init__(self, **kwargs):
        
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False, 
            show_coordinates=True,  # include coordinate labels
            leave_ghost_vectors=False,
            include_foreground_plane=True,  # includes grid lines 
            include_background_plane=True,  # includes numbers 
            *kwargs
        )
        
  def construct(self):
    add_standard_basis(self=self)


class nonStandardBasis(LinearTransformationScene):
  
  def __init__(self, **kwargs):
        
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False, 
            show_coordinates=True,  # include coordinate labels
            leave_ghost_vectors=False,
            include_foreground_plane=True,  # includes grid lines 
            include_background_plane=True,  # includes numbers 
            *kwargs
        )
        
  def construct(self):

    add_standard_basis(self=self, remove_elements_at_end=True)

    add_non_standard_basis(self=self)
   

class tomBasisVectors(LinearTransformationScene):
  
  def __init__(self, **kwargs):
        
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False, 
            show_coordinates=True,  # include coordinate labels
            leave_ghost_vectors=False,
            include_foreground_plane=True,  # includes grid lines 
            include_background_plane=True,  # includes numbers 
            *kwargs
        )
        
  def construct(self):
    
    # region Scene 1: Basis vector x
    basis_x = Vector([1, 0], color="#4EF716")
    
    # add label for basis_x 
    label_x = MathTex(r'\mathbf{b}_{\mathbf{t_x}}', color="#4EF716", font_size=35)  
    label_x.align_to(basis_x, UL)
    label_x.shift(0.25 * DOWN + 0.45 * RIGHT)

    self.add_vector(basis_x)
    self.wait(0.5)
    # endregion 

    # region Scene 2: First column with vector label x
    text = (MathTex(
            r"\begin{array}{c}\begin{matrix}" + 
            r"\hspace{1cm} \mathbf{b_{t_x}}\end{matrix} \\ " +
            r" \mathbf{B}_t = \begin{bmatrix} 1 \\ \\" + 
            r"0  \end{bmatrix} \end{array}",font_size=35). 
            to_edge(UL))
                     
      
    # add custom colouring
    # self.add(index_labels(text[0]))  # help determine what index values to colour
    text[0][0:3].set_color("#4EF716")
    text.add_background_rectangle()

    # Display the initial text
    show_text = [Create(text), Write(label_x)]
    self.play(*show_text)
    self.wait(0.5)
    # endregion 

    # region Scene 3: Basis vector y
    basis_y = Vector([0, 1], color="#FB699D")
        
    # add label for basis_y
    label_y = MathTex(r'\mathbf{b}_{\mathbf{t_y}}', color="#FB699D", font_size=35)  
    label_y.align_to(basis_y, UR)
    label_y.shift(0.3 * LEFT)

    self.add_vector(basis_y)
    self.wait(0.5)
    # endregion 

    # region Scene 4: Second column with vector label y

    text2 = (MathTex(r"\begin{array}{c}\begin{matrix}" + 
                       r"\hspace{1cm} \mathbf{b_{t_x}} & \mathbf{b_{t_y}} \end{matrix} \\ " +
                       r" \mathbf{B}_t = \begin{bmatrix} 1 & \quad 0 \\ \\" + 
                       r"0 & \quad 1 \end{bmatrix} \end{array}",font_size=35)
                       .to_edge(UL))

    # add custom colouring
    # self.add(index_labels(text[0]))  # help determine what index values to colour
    text2[0][0:3].set_color("#4EF716")
    text2[0][3:6].set_color("#FB699D")
    text2.add_background_rectangle()
    
    self.play(Transform(text, text2), Write(label_y))
    self.wait(0.5)
    # endregion

    # region Scene 5: Add x component of vector h

    vec_basis_x = Vector(color="#4EF716")
    vec_basis_x.put_start_and_end_on(start=[1,0, 0], end=[-2, 0, 0])
    h_coord_x = MathTex(r'-2\mathbf{b}_{\mathbf{t_x}}', color="#4EF716", font_size=35)  
    h_coord_x.align_to(vec_basis_x, UL)
    h_coord_x.shift(0.35 * UP)


    self.add_vector(vec_basis_x) 
    self.wait()
    self.play(Write(h_coord_x))

    # endregion 
      
    # region Scene 6: Add y component of vector h and move to tip of x component
    vec_basis_y = Vector(color="#FB699D")
    vec_basis_y.put_start_and_end_on(start=[0,1, 0], end=[0, -1, 0])
    h_coord_y = MathTex(r'-1\mathbf{b}_{\mathbf{t_y}}', color="#FB699D", font_size=35)  
    h_coord_y.align_to(vec_basis_y, DL)
    h_coord_y.shift(1 * LEFT + 0.2 * UP)

    # Create a VGroup to include both the vector and its label
    vector_with_label = VGroup(vec_basis_y, h_coord_y)

    self.add_vector(vec_basis_y) 
    self.wait()
    self.play(Write(h_coord_y))

    vec_basis_y.put_start_and_end_on(start=ORIGIN, end=[0, -1, 0])
    animation = ApplyMethod(vector_with_label.shift, [-2, 0, 0])
    self.play(animation)

    # endregion 

    # region Scene 7: Add vector h 
    vec_h = Vector([-2, -1], color='#d170c7')
    label_vec_h = MathTex(r'\mathbf{h_t} = [-2, -1]', color='#d170c7', font_size=35)
    label_vec_h.add_background_rectangle()
    label_vec_h.move_to([-1.5, -1.5, 0])
    # label_vec_h.next_to(vec_h, direction=[-1, -1, 0], buff=0.1)

    self.add_vector(vec_h)
    self.play(Write(label_vec_h))
    self.wait(1.5)
    
    # endregion


class sarahBasisVectors(LinearTransformationScene):
  
  def __init__(self, **kwargs):
        
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False, 
            show_coordinates=True,  # include coordinate labels
            leave_ghost_vectors=False,
            include_foreground_plane=True,  # includes grid lines 
            include_background_plane=True,  # includes numbers 
            *kwargs
        )
        
  def construct(self):
    
    # region Scene 1: Basis vector x
    basis_x = Vector([1, -1], color="#4EF716")
    
    # add label for basis_x 
    label_x = MathTex(r'\mathbf{b}_{\mathbf{s_x}}', color="#4EF716", font_size=35)  
    label_x.align_to(basis_x, UL)
    label_x.shift(0.5 * DOWN + 1 * RIGHT)

    self.add_vector(basis_x)
    self.wait(0.5)
    # endregion 

    # region Scene 2: First column with vector label x
    text = (MathTex(
            r"\begin{array}{c}\begin{matrix}" + 
            r"\hspace{1cm} \mathbf{b_{s_x}}\end{matrix} \\ " +
            r" \mathbf{B}_s = \begin{bmatrix} 1 \\ \\" + 
            r"-1 \end{bmatrix} \end{array}",font_size=35). 
            to_edge(UL))
                     
      
    # add custom colouring
    # self.add(index_labels(text[0]))  # help determine what index values to colour
    text[0][0:3].set_color("#4EF716")
    text.add_background_rectangle()

    # Display the initial text
    show_text = [Create(text), Write(label_x)]
    self.play(*show_text)
    self.wait(0.5)
    # endregion 

    # region Scene 3: Basis vector y
    basis_y = Vector([1, 1], color="#FB699D")
        
    # add label for basis_y
    label_y = MathTex(r'\mathbf{b}_{\mathbf{s_y}}', color="#FB699D", font_size=35)  
    label_y.align_to(basis_y, UR)
    label_y.shift(0.6 * RIGHT)

    self.add_vector(basis_y)
    self.wait(0.5)
    # endregion 

    # region Scene 4: Second column with vector label y

    text2 = (MathTex(r"\begin{array}{c}\begin{matrix}" + 
                       r"\hspace{1cm} \mathbf{b_{e_x}} & \mathbf{b_{_y}} \end{matrix} \\ " +
                       r" \mathbf{B}_e = \begin{bmatrix} 1 & \quad 1 \\ \\" + 
                       r"-1 & \quad 1 \end{bmatrix} \end{array}",font_size=35)
                       .to_edge(UL))

    # add custom colouring
    # self.add(index_labels(text[0]))  # help determine what index values to colour
    text2[0][0:3].set_color("#4EF716")
    text2[0][3:6].set_color("#FB699D")
    text2.add_background_rectangle()
    
    self.play(Transform(text, text2), Write(label_y))
    self.wait(0.5)
    # endregion

    # region Scene 5: Add x component of vector h
    start_x = np.array([1, -1, 0])
    end_x = -0.5*start_x

    vec_basis_x = Vector(color="#4EF716")
    vec_basis_x.put_start_and_end_on(start=start_x, end=end_x)
    h_coord_x = MathTex(r'-0.5\mathbf{b}_{\mathbf{s_x}}', color="#4EF716", font_size=35)  
    h_coord_x.align_to(vec_basis_x, UL)
    h_coord_x.shift(0.1* DOWN + 1.2 * LEFT )


    self.add_vector(vec_basis_x) 
    self.wait()
    self.play(Write(h_coord_x))

    # endregion 
      
    # region Scene 6: Add y component of vector h and move start of 
    # x component to to tip of y component
    start_y = np.array([1, 1, 0])
    end_y = -1.5*start_y

    vec_basis_y = Vector(color="#FB699D")
    vec_basis_y.put_start_and_end_on(start=start_y, end=end_y)
    h_coord_y = MathTex(r'-1.5\mathbf{b}_{\mathbf{s_y}}', color="#FB699D", font_size=35)  
    h_coord_y.add_background_rectangle()
    h_coord_y.align_to(vec_basis_y, DL)
    h_coord_y.shift(0.1* RIGHT + 0.4*DOWN)

    # Create a VGroup to include both the vector and its label
    vector_with_label = VGroup(vec_basis_x, h_coord_x)

    self.add_vector(vec_basis_y) 
    self.wait()
    self.play(Write(h_coord_y))

    vec_basis_x.put_start_and_end_on(start=ORIGIN, end=end_x)
    animation = ApplyMethod(vector_with_label.shift, end_y)
    self.play(animation)

    # endregion 

    # region Scene 7: Add vector h 
    vec_h = Vector([-2, -1], color='#d170c7')
    label_vec_h = MathTex(r'\mathbf{h_s} = [-0.5, -1.5]', color='#d170c7', font_size=35)
    label_vec_h.add_background_rectangle()
    label_vec_h.move_to([-3.4, -0.5, 0])
    # label_vec_h.next_to(vec_h, direction=[-1, -1, 0], buff=0.1)

    self.add_vector(vec_h)
    self.play(Write(label_vec_h))
    self.wait(1.5)
    
    # endregion


class tomCoordTransformation(LinearTransformationScene):

    def __init__(self, **kwargs):
        
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False, 
            show_coordinates=True,  # include coordinate labels
            leave_ghost_vectors=False,
            include_foreground_plane=True,  # includes grid lines 
            include_background_plane=True,  # includes numbers 
            *kwargs
        )

 
    def construct(self):
        

        # vector g_e
        # region Scene 1: Basis vector x
        vector_g_e = Vector([1.5, 0.5], color="#4EF716")

        # add label for vector_g_e 
        # # label_x = MathTex(r'\mathbf{g}_{\mathbf{x}}', color="#4EF716", font_size=35)  
        # label_x.align_to(vector_g_e, UL)
        # label_x.shift(0.25 * DOWN + 0.3 * RIGHT)

        self.add_vector(vector_g_e)
        self.wait(0.5)  
        self.moving_mobjects = []

        # non-standard basis
        non_standard_basis = np.array([[1, 1], 
                              [0, 2]])

        
        # apply transformation 
        self.apply_matrix(non_standard_basis)
        self.wait()        


class transformationExample(LinearTransformationScene):

    def __init__(self, **kwargs):
        
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False, 
            show_coordinates=True,  # include coordinate labels
            leave_ghost_vectors=True,
            include_foreground_plane=True,  # includes grid lines 
            include_background_plane=True,  # includes numbers 
            *kwargs
        )

 
    def construct(self):
        
        # region Scene 1: Vector in standard basis 
        vec_ge = Vector([2, 1], color='#d170c7')
        vec_ge_label = MathTex(r'\mathbf{g_e} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}', 
                               color="#d170c7", 
                               font_size=35)
        # vec_ge_label.align_to(vec_ge, UR)
        vec_ge_label.add_background_rectangle()
        vec_ge_label.move_to([4.75, 1.75, 0])  

        self.add_vector(vec_ge)
        show_vec_ge_label = [Write(vec_ge_label)]
        self.play(*show_vec_ge_label)
        self.wait(0.75)

        # endregion 

        # region Scene 2: Basis vector x
        basis_x = Vector([1, 0], color="#4EF716")

        # add label for basis_x 
        label_x = MathTex(r'\mathbf{b}_{\mathbf{e_x}}', color="#4EF716", font_size=35)  
        label_x.align_to(basis_x, UL)
        label_x.shift(0.25 * DOWN + 0.45 * RIGHT)

        self.add_vector(basis_x)
        self.wait(0.5)
        self.play(Write(label_x))
        self.wait(0.5)

        # endregion

        # region Scene 3: Basis vector y
        basis_y = Vector([0, 1], color="#FB699D")

        # add label for basis_y
        label_y = MathTex(r'\mathbf{b}_{\mathbf{e_y}}', color="#FB699D", font_size=35)  
        label_y.align_to(basis_y, UR)
        label_y.shift(0.3 * LEFT)

        self.add_vector(basis_y)
        self.play(Write(label_y))

        self.wait(0.5)
        # endregion 

        # region Scene 4: Add equation 
        equation = (MathTex(r"\mathbf{g_n} = \mathbf{B_n}^{-1} \mathbf{g_e}",
                            font_size=35)
                            .to_edge(UR))
        equation.add_background_rectangle()

        # Display the initial equation
        show_equation = [Create(equation)]
        self.play(*show_equation)
        self.wait(0.5)
        self.moving_mobjects = []

        # endregion    

        # region Scene 5: Show inverse matrix text
        # add basis vectors of 
        inverse_text = (MathTex(r"\mathbf{B}_n^{-1} = \begin{bmatrix} 1 & -0.5 \\ \\" + 
                       r"0 &  0.5 \end{bmatrix}",
                       font_size=35)
                       .to_edge(UL))

        # add custom colouring
        # self.add(index_labels(text[0]))  # help determine what index values to colour
        inverse_text.add_background_rectangle()

        # Display the initial inverse_text
        show_inverse_text = [Create(inverse_text)]
        self.play(*show_inverse_text)
        self.wait(0.5)
        self.moving_mobjects = []

        # endregion

        # region Scene 6: Apply inverse of non-standard basis and transformed vector label
        # non-standard basis
        non_standard_basis = np.array([[1, 1], 
                                        [0, 2]])

        self.add(vec_ge.copy())
        
        # apply transformation 
        self.apply_inverse(non_standard_basis)
        vec_ge.set_color('#f5dc0e')


        # show label of vector g_n
        vec_gn_label = MathTex(r'\mathbf{g_n} = \begin{bmatrix} 1.5 \\ 0.5 \end{bmatrix}', 
                               color="#f5dc0e", 
                               font_size=35)
        
        vec_gn_label.add_background_rectangle()
        vec_gn_label.move_to([4.75, 0.5, 0])  
        
        show_vec_gn_label = [Write(vec_gn_label)]
        self.play(*show_vec_gn_label)
        self.wait(0.75)

        # endregion 

        # region Scene 7: Change basis x label 
        inverse_text_labels_x = (MathTex(r"\begin{array}{c}\begin{matrix}" + 
                           r"\hspace{1cm} \mathbf{b^{-1}_{n_x}}  \quad \hspace{0.6cm} \end{matrix} \\ " +
                           r" \mathbf{B}^{-1}_n = \begin{bmatrix} 1 & \quad -0.5 \\ \\" + 
                           r"0 & \quad 0.5 \end{bmatrix} \end{array}", font_size=35)
                           .to_edge(UL))
        
        # add custom colouring
        # self.add(index_labels(text[0]))  # help determine what index values to colour
        inverse_text_labels_x[0][0:5].set_color("#4EF716")
        inverse_text_labels_x.add_background_rectangle()
        
        # label for first basis vector of inverse 
        inverse_label_x = MathTex(r'\mathbf{b^{-1}_{n_x}}', color="#4EF716", font_size=35)  
        inverse_label_x.add_background_rectangle()
        inverse_label_x.align_to(basis_x, UL)
        inverse_label_x.shift(0.3 * DOWN + 0.4* RIGHT)

        self.play(Transform(inverse_text, inverse_text_labels_x), 
                  Transform(label_x, inverse_label_x))

        self.wait(0.5)
        self.moving_mobjects = []  

        # endregion 

        # region Scene 8: Change basis y label
        inverse_text_labels = (MathTex(r"\begin{array}{c}\begin{matrix}" + 
           r"\hspace{1cm} \mathbf{b^{-1}_{n_x}} & \mathbf{b^{-1}_{n_y}} \end{matrix} \\ " +
           r" \mathbf{B}^{-1}_n = \begin{bmatrix} 1 & \quad -0.5 \\ \\" + 
           r"0 & \quad 0.5 \end{bmatrix} \end{array}", font_size=35)
           .to_edge(UL))
        
        inverse_text_labels[0][0:5].set_color("#4EF716")
        inverse_text_labels[0][5:10].set_color("#FB699D")
        inverse_text_labels.add_background_rectangle()

        # label for second basis vector of inverse 
        inverse_label_y = MathTex(r'\mathbf{b^{-1}_{n_y}}', color="#FB699D", font_size=35)  
        inverse_label_y.add_background_rectangle()
        inverse_label_y.align_to(basis_y, UR)
        inverse_label_y.shift(0.5 * LEFT + 0.3*UP)

        self.play(Transform(inverse_text_labels_x, inverse_text_labels), 
                  Transform(label_y, inverse_label_y))
    
        # endregion


class diagonalMatrix(LinearTransformationScene):

    def __init__(self, **kwargs):
        
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False, 
            show_coordinates=True,  # include coordinate labels
            leave_ghost_vectors=True,
            include_foreground_plane=True,  # includes grid lines 
            include_background_plane=True,  # includes numbers 
            *kwargs
        )

 
    def construct(self):
        
        # region Scene 1: Vector in standard basis 
        vec_ge = Vector([2, 1], color='#d170c7')
        vec_ge_label = MathTex(r'\mathbf{g_e} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}', 
                               color="#d170c7", 
                               font_size=35)
        # vec_ge_label.align_to(vec_ge, UR)
        vec_ge_label.add_background_rectangle()
        vec_ge_label.move_to([5.25, 1.75, 0])  

        self.add_vector(vec_ge)
        show_vec_ge_label = [Write(vec_ge_label)]
        self.play(*show_vec_ge_label)
        self.wait(0.75)
        
        # endregion 

        # region Scene 2: Basis vector x
        basis_x = Vector([1, 0], color="#4EF716")

        # add label for basis_x 
        label_x = MathTex(r'\mathbf{b}_{\mathbf{e_x}}', color="#4EF716", font_size=35)  
        label_x.align_to(basis_x, UL)
        label_x.shift(0.25 * DOWN + 0.45 * RIGHT)

        self.add_vector(basis_x)
        self.wait(0.5)
        self.play(Write(label_x))
        self.wait(0.5)

        # endregion

        # region Scene 3: Basis vector y
        basis_y = Vector([0, 1], color="#FB699D")

        # add label for basis_y
        label_y = MathTex(r'\mathbf{b}_{\mathbf{e_y}}', color="#FB699D", font_size=35)  
        label_y.align_to(basis_y, UR)
        label_y.shift(0.3 * LEFT)

        self.add_vector(basis_y)
        self.play(Write(label_y))

        self.wait(0.5)
        # endregion 

        # region Scene 4: Add equation 
        equation = (MathTex(r"\mathbf{g_d} = \mathbf{D} \mathbf{g_e}",
                            font_size=35)
                            .to_edge(UR))
        equation.add_background_rectangle()

        # Display the initial equation
        show_equation = [Create(equation)]
        self.play(*show_equation)
        self.wait(0.5)
        self.moving_mobjects = []

        # endregion    

        # region Scene 5: Show inverse matrix text
        # add basis vectors of 
        inverse_text = (MathTex(r"\mathbf{D} = \begin{bmatrix} 2 & 0 \\ \\" + 
                       r"0 &  3 \end{bmatrix}",
                       font_size=35)
                       .to_edge(UL))

        # add custom colouring
        # self.add(index_labels(text[0]))  # help determine what index values to colour
        inverse_text.add_background_rectangle()

        # Display the initial inverse_text
        show_inverse_text = [Create(inverse_text)]
        self.play(*show_inverse_text)
        self.wait(0.5)
        self.moving_mobjects = []

        # endregion
        
        # region Scene 6: Apply matrix and transformed vector label
        # non-standard basis
        non_standard_basis = np.array([[2, 0], 
                                        [0, 3]])

        self.add(vec_ge.copy())
        
        # apply transformation 
        self.apply_matrix(non_standard_basis)

        # change vector color
        vec_ge.set_color('#f5dc0e')

        # show label of vector g_n
        vec_gn_label = MathTex(r'\mathbf{g_d} = \begin{bmatrix} 4 \\ 3 \end{bmatrix}', 
                               color="#f5dc0e", 
                               font_size=35)
        
        vec_gn_label.add_background_rectangle()
        vec_gn_label.move_to([5.25, 0.5, 0])  
        
        show_vec_gn_label = [Write(vec_gn_label)]
        self.play(*show_vec_gn_label)
        self.wait(0.75)

        # endregion 
       
        # region Scene 7: Change basis x label 
        inverse_text_labels_x = (MathTex(r"\begin{array}{c}\begin{matrix}" + 
                           r"\hspace{1cm} \mathbf{b_{d_x}}  \quad \hspace{0.6cm} \end{matrix} \\ " +
                           r" \mathbf{D} = \begin{bmatrix} 2 & \quad 0 \\ \\" + 
                           r"0 & \quad 3 \end{bmatrix} \end{array}", font_size=35)
                           .to_edge(UL))
        
        # add custom colouring
        # self.add(index_labels(text[0]))  # help determine what index values to colour
        inverse_text_labels_x[0][0:3].set_color("#4EF716")
        inverse_text_labels_x.add_background_rectangle()
        
        # label for first basis vector of inverse 
        inverse_label_x = MathTex(r'\mathbf{b_{d_x}}', color="#4EF716", font_size=35)  
        inverse_label_x.add_background_rectangle()
        inverse_label_x.align_to(basis_x, UR)
        inverse_label_x.shift(0.3 * DOWN + 0.4* RIGHT)

        self.play(Transform(inverse_text, inverse_text_labels_x), 
                  Transform(label_x, inverse_label_x))

        self.wait(0.5)
        self.moving_mobjects = []  

        # endregion 

        # region Scene 8: Change basis y label
        inverse_text_labels = (MathTex(r"\begin{array}{c}\begin{matrix}" + 
           r"\hspace{1cm} \mathbf{b_{d_x}} & \mathbf{b_{d_y}}  \end{matrix} \\ " +
           r" \mathbf{D} = \begin{bmatrix} 2 & \quad 0 \\ \\" + 
           r"0 & \quad 3 \end{bmatrix} \end{array}", font_size=35)
           .to_edge(UL))
        
        inverse_text_labels[0][0:3].set_color("#4EF716")
        inverse_text_labels[0][3:6].set_color("#FB699D")
        inverse_text_labels.add_background_rectangle()

        # label for second basis vector of inverse 
        inverse_label_y = MathTex(r'\mathbf{b_{d_y}}', color="#FB699D", font_size=35)  
        inverse_label_y.add_background_rectangle()
        inverse_label_y.align_to(basis_y, UR)
        inverse_label_y.shift(0.5 * LEFT + 0.3*UP)

        self.play(Transform(inverse_text_labels_x, inverse_text_labels), 
                  Transform(label_y, inverse_label_y))
        
        self.wait(0.5)
    
        # endregion


class orthonormalMatrix(LinearTransformationScene):

    def __init__(self, **kwargs):
        
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False, 
            show_coordinates=True,  # include coordinate labels
            leave_ghost_vectors=True,
            include_foreground_plane=True,  # includes grid lines 
            include_background_plane=True,  # includes numbers 
            *kwargs
        )

 
    def construct(self):
        
        # region Scene 1: Vector in standard basis 
        vec_ge = Vector([2, 1], color='#d170c7')
        vec_ge_label = MathTex(r'\mathbf{g_e} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}', 
                               color="#d170c7", 
                               font_size=35)
        # vec_ge_label.align_to(vec_ge, UR)
        vec_ge_label.add_background_rectangle()
        vec_ge_label.move_to([5.25, 1.75, 0])  

        self.add_vector(vec_ge)
        show_vec_ge_label = [Write(vec_ge_label)]
        self.play(*show_vec_ge_label)
        self.wait(0.75)
        
        # endregion 

        # region Scene 2: Basis vector x
        basis_x = Vector([1, 0], color="#4EF716")

        # add label for basis_x 
        label_x = MathTex(r'\mathbf{b}_{\mathbf{e_x}}', color="#4EF716", font_size=35)  
        label_x.align_to(basis_x, UL)
        label_x.shift(0.25 * DOWN + 0.45 * RIGHT)

        self.add_vector(basis_x)
        self.wait(0.5)
        self.play(Write(label_x))
        self.wait(0.5)

        # endregion

        # region Scene 3: Basis vector y
        basis_y = Vector([0, 1], color="#FB699D")

        # add label for basis_y
        label_y = MathTex(r'\mathbf{b}_{\mathbf{e_y}}', color="#FB699D", font_size=35)  
        label_y.align_to(basis_y, UR)
        label_y.shift(0.3 * LEFT)

        self.add_vector(basis_y)
        self.play(Write(label_y))

        self.wait(0.5)
        # endregion 

        # region Scene 4: Add equation 
        equation = (MathTex(r"\mathbf{g_q} = \mathbf{Q} \mathbf{g_e}",
                            font_size=35)
                            .to_edge(UR))
        equation.add_background_rectangle()

        # Display the initial equation
        show_equation = [Create(equation)]
        self.play(*show_equation)
        self.wait(0.5)
        self.moving_mobjects = []

        # endregion    

        # region Scene 5: Show orthonormal matrix text
        # add basis vectors of 
        orthonormal_text = (MathTex(r"\mathbf{Q} = \begin{bmatrix} \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\ \\" + 
                       r"-\frac{\sqrt{2}}{2} &  \frac{\sqrt{2}}{2} \end{bmatrix}",
                       font_size=35)
                       .to_edge(UL))

        # add custom colouring
        # self.add(index_labels(text[0]))  # help determine what index values to colour
        orthonormal_text.add_background_rectangle()

        # Display the initial orthonormal_text
        show_orthonormal_text = [Create(orthonormal_text)]
        self.play(*show_orthonormal_text)
        self.wait(0.5)
        self.moving_mobjects = []

        # endregion
        
        # region Scene 6: Apply matrix and transformed vector label
        # non-standard basis
        non_standard_basis = np.array([[np.sqrt(2)/2, np.sqrt(2)/2], 
                                        [-np.sqrt(2)/2, np.sqrt(2)/2]])

        self.add(vec_ge.copy())
        
        # apply transformation 
        self.apply_matrix(non_standard_basis)

        # change vector color
        vec_ge.set_color('#f5dc0e')

        # show label of vector g_n
        vec_gq_label = MathTex(r'\mathbf{g_q} = \begin{bmatrix} 3 \frac{\sqrt{2}}{2} \\ - \frac{\sqrt{2}}{2} \end{bmatrix}', 
                               color="#f5dc0e", 
                               font_size=35)
        
        vec_gq_label.add_background_rectangle()
        vec_gq_label.move_to([5.25, 0.5, 0])  
        
        show_vec_gq_label = [Write(vec_gq_label)]
        self.play(*show_vec_gq_label)
        self.wait(0.75)

        # endregion 
        
        # region Scene 7: Change basis x label 
        orthonormal_text_labels_x = (MathTex(
            r"\begin{array}{c}\begin{matrix}" + 
            r"\hspace{1cm} \mathbf{b_{q_x}}  \quad \hspace{0.6cm} \end{matrix} \\ " +
            r"\mathbf{Q} = \begin{bmatrix} \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\ \\" + 
            r"-\frac{\sqrt{2}}{2} &  \frac{\sqrt{2}}{2} \end{bmatrix} \end{array}", font_size=35).to_edge(UL))
        
        # add custom colouring
        # self.add(index_labels(text[0]))  # help determine what index values to colour
        orthonormal_text_labels_x[0][0:3].set_color("#4EF716")
        orthonormal_text_labels_x.add_background_rectangle()
        
        # label for first basis vector of orthonormal 
        orthonormal_label_x = MathTex(r'\mathbf{b_{q_x}}', color="#4EF716", font_size=35)  
        orthonormal_label_x.add_background_rectangle()
        orthonormal_label_x.align_to(basis_x, UR)
        orthonormal_label_x.shift(0.4 * DOWN + 0.3* LEFT)

        self.play(Transform(orthonormal_text, orthonormal_text_labels_x), 
                  Transform(label_x, orthonormal_label_x))

        self.wait(0.5)
        self.moving_mobjects = []  

        # endregion 

        # region Scene 8: Change basis y label
        orthonormal_text_labels = (MathTex(r"\begin{array}{c}\begin{matrix}" + 
           r"\hspace{1cm} \mathbf{b_{q_x}} & \mathbf{b_{q_y}}  \end{matrix} \\ " +
           r"\mathbf{Q} = \begin{bmatrix} \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\ \\" + 
            r"-\frac{\sqrt{2}}{2} &  \frac{\sqrt{2}}{2} \end{bmatrix} \end{array}", font_size=35).to_edge(UL))
        
        orthonormal_text_labels[0][0:3].set_color("#4EF716")
        orthonormal_text_labels[0][3:6].set_color("#FB699D")
        orthonormal_text_labels.add_background_rectangle()

        # label for second basis vector of orthonormal 
        orthonormal_label_y = MathTex(r'\mathbf{b_{q_y}}', color="#FB699D", font_size=35)  
        orthonormal_label_y.add_background_rectangle()
        orthonormal_label_y.align_to(basis_y, UR)
        orthonormal_label_y.shift(0.3 * LEFT + 0.4*UP)

        self.play(Transform(orthonormal_text_labels_x, orthonormal_text_labels), 
                  Transform(label_y, orthonormal_label_y))
        
        self.wait(1)
    
        # endregion


class inverseMatrix(LinearTransformationScene):

    def __init__(self, **kwargs):
        
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False, 
            show_coordinates=True,  # include coordinate labels
            leave_ghost_vectors=True,
            include_foreground_plane=True,  # includes grid lines 
            include_background_plane=False,  # includes numbers 
            *kwargs
        )

 
    def construct(self):
        
        # region Scene 1: Vector in standard basis 
        vec_ge = Vector([2, 1], color='#d170c7')
        vec_ge_label = MathTex(r'\mathbf{g_e} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}', 
                               color="#d170c7", 
                               font_size=35)
        # vec_ge_label.align_to(vec_ge, UR)
        vec_ge_label.add_background_rectangle()
        vec_ge_label.move_to([5.25, 1.75, 0])  

        self.add_vector(vec_ge)
        show_vec_ge_label = [Write(vec_ge_label)]
        self.play(*show_vec_ge_label)
        self.wait(0.75)
        
        # endregion 

        # region Scene 2: Basis vector x
        basis_x = Vector([1, 0], color="#4EF716")

        # add label for basis_x 
        label_x = MathTex(r'\mathbf{b}_{\mathbf{e_x}}', color="#4EF716", font_size=35)  
        label_x.align_to(basis_x, UL)
        label_x.shift(0.25 * DOWN + 0.45 * RIGHT)

        self.add_vector(basis_x)
        self.wait(0.5)
        self.play(Write(label_x))
        self.wait(0.5)

        # endregion

        # region Scene 3: Basis vector y
        basis_y = Vector([0, 1], color="#FB699D")

        # add label for basis_y
        label_y = MathTex(r'\mathbf{b}_{\mathbf{e_y}}', color="#FB699D", font_size=35)  
        label_y.align_to(basis_y, UR)
        label_y.shift(0.3 * LEFT)

        self.add_vector(basis_y)
        self.play(Write(label_y))

        self.wait(0.5)
        # endregion 

        # region Scene 4: Add equation 
        equation = (MathTex(r"\mathbf{g_q} = \mathbf{Q} \mathbf{g_e}",
                            font_size=35)
                            .to_edge(UR))
        equation.add_background_rectangle()

        # Display the initial equation
        show_equation = [Create(equation)]
        self.play(*show_equation)
        self.wait(0.5)
        self.moving_mobjects = []

        # endregion    

        # region Scene 5: Show orthonormal matrix text
        # add basis vectors of 
        orthonormal_text = (MathTex(r"\mathbf{Q} = \begin{bmatrix} \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\ \\" + 
                       r"-\frac{\sqrt{2}}{2} &  \frac{\sqrt{2}}{2} \end{bmatrix}",
                       font_size=35)
                       .to_edge(UL))

        # add custom colouring
        # self.add(index_labels(text[0]))  # help determine what index values to colour
        orthonormal_text.add_background_rectangle()

        # Display the initial orthonormal_text
        show_orthonormal_text = [Create(orthonormal_text)]
        self.play(*show_orthonormal_text)
        self.wait(0.5)
        self.moving_mobjects = []

        # endregion
        
        # region Scene 6: Apply matrix and transformed vector label
        # non-standard basis
        non_standard_basis = np.array([[np.sqrt(2)/2, np.sqrt(2)/2], 
                                        [-np.sqrt(2)/2, np.sqrt(2)/2]])

        self.add(vec_ge.copy())
        
        # apply transformation 
        self.apply_matrix(non_standard_basis)

        # change vector color
        vec_ge.set_color('#f5dc0e')

        # show label of vector g_n
        vec_gq_label = MathTex(r'\mathbf{g_q} = \begin{bmatrix} 3 \frac{\sqrt{2}}{2} \\ - \frac{\sqrt{2}}{2} \end{bmatrix}', 
                               color="#f5dc0e", 
                               font_size=35)
        
        vec_gq_label.add_background_rectangle()
        vec_gq_label.move_to([5.25, 0.5, 0])  
        
        show_vec_gq_label = [Write(vec_gq_label)]
        self.play(*show_vec_gq_label)
        self.wait(0.75)

        # endregion 
        
        # region Scene 7: Change basis x label 
        orthonormal_text_labels_x = (MathTex(
            r"\begin{array}{c}\begin{matrix}" + 
            r"\hspace{1cm} \mathbf{b_{q_x}}  \quad \hspace{0.6cm} \end{matrix} \\ " +
            r"\mathbf{Q} = \begin{bmatrix} \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\ \\" + 
            r"-\frac{\sqrt{2}}{2} &  \frac{\sqrt{2}}{2} \end{bmatrix} \end{array}", font_size=35).to_edge(UL))
        
        # add custom colouring
        # self.add(index_labels(text[0]))  # help determine what index values to colour
        orthonormal_text_labels_x[0][0:3].set_color("#4EF716")
        orthonormal_text_labels_x.add_background_rectangle()
        
        # label for first basis vector of orthonormal 
        orthonormal_label_x = MathTex(r'\mathbf{b_{q_x}}', color="#4EF716", font_size=35)  
        orthonormal_label_x.add_background_rectangle()
        orthonormal_label_x.align_to(basis_x, UR)
        orthonormal_label_x.shift(0.4 * DOWN + 0.3* LEFT)

        self.play(Transform(orthonormal_text, orthonormal_text_labels_x), 
                  Transform(label_x, orthonormal_label_x))

        self.wait(0.5)
        self.moving_mobjects = []  

        # endregion 

        # region Scene 8: Change basis y label
        orthonormal_text_labels = (MathTex(r"\begin{array}{c}\begin{matrix}" + 
           r"\hspace{1cm} \mathbf{b_{q_x}} & \mathbf{b_{q_y}}  \end{matrix} \\ " +
           r"\mathbf{Q} = \begin{bmatrix} \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\ \\" + 
            r"-\frac{\sqrt{2}}{2} &  \frac{\sqrt{2}}{2} \end{bmatrix} \end{array}", font_size=35).to_edge(UL))
        
        orthonormal_text_labels[0][0:3].set_color("#4EF716")
        orthonormal_text_labels[0][3:6].set_color("#FB699D")
        orthonormal_text_labels.add_background_rectangle()

        # label for second basis vector of orthonormal 
        orthonormal_label_y = MathTex(r'\mathbf{b_{q_y}}', color="#FB699D", font_size=35)  
        orthonormal_label_y.add_background_rectangle()
        orthonormal_label_y.align_to(basis_y, UR)
        orthonormal_label_y.shift(0.3 * LEFT + 0.4*UP)

        self.play(Transform(orthonormal_text_labels_x, orthonormal_text_labels), 
                  Transform(label_y, orthonormal_label_y))
        
        self.wait(1)
    
        # endregion

        # region Scene 9: Show inverse equation
        inv_equation = (MathTex(r"\mathbf{g_e} = \mathbf{Q}^{-1} \mathbf{g_q}",
                    font_size=35)
                    .to_edge(UR))
        
        inv_equation.shift(0.7*DOWN)
        
        inv_equation.add_background_rectangle()

        # Display the initial inv_equation
        show_inv_equation = [Create(inv_equation)]
        self.play(*show_inv_equation)
        self.wait(0.5)
        self.moving_mobjects = []

        # endregion  

        # region Scene 10: Apply inverse of matrix 
                  
        # apply transformation 
        self.apply_inverse(non_standard_basis)

        # change vector color
        vec_ge.set_color('#d170c7')
        
        self.moving_mobjects = []  

        # endregion

        # region Scene 11: Reverse basis vector x 
        # add label for basis_x 
        label_x_orig= MathTex(r'\mathbf{b}_{\mathbf{e_x}}', color="#4EF716", font_size=35)  
        label_x_orig.align_to(basis_x, UL)
        label_x_orig.shift(0.25 * DOWN + 0.45 * RIGHT)

        self.play(Transform(label_x, label_x_orig).set_run_time(1))
        self.moving_mobjects = []  
        
        # add label for basis_y
        label_y_orig = MathTex(r'\mathbf{b}_{\mathbf{e_y}}', color="#FB699D", font_size=35)  
        label_y_orig.align_to(basis_y, UR)
        label_y_orig.shift(0.3 * LEFT)

        self.play(Transform(label_y, label_y_orig))

        # endregion


class rectangularMatrix(LinearTransformationScene):

    def __init__(self, **kwargs):
        
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False, 
            show_coordinates=True,  # include coordinate labels
            leave_ghost_vectors=False,
            include_foreground_plane=True,  # includes grid lines 
            include_background_plane=True,  # includes numbers 
            *kwargs
        )

    def construct(self):

        # region Scene 1: Vector in standard basis 
        vec_ge = Vector([2, 1], color='#d170c7')
        vec_ge_label = MathTex(r'\mathbf{g_e} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}', 
                               color="#d170c7", 
                               font_size=35)
        # vec_ge_label.align_to(vec_ge, UR)
        vec_ge_label.add_background_rectangle()
        vec_ge_label.move_to([5.25, 1.75, 0])  

        self.add_vector(vec_ge)
        show_vec_ge_label = [Write(vec_ge_label)]
        self.play(*show_vec_ge_label)
        self.wait(0.75)
        
        # endregion 

        # region Scene 2: Basis vector x
        basis_x = Vector([1, 0], color="#4EF716")

        # add label for basis_x 
        label_x = MathTex(r'\mathbf{b}_{\mathbf{e_x}}', color="#4EF716", font_size=35)  
        label_x.align_to(basis_x, UL)
        label_x.shift(0.25 * DOWN + 0.45 * RIGHT)

        self.add_vector(basis_x)
        self.wait(0.5)
        self.play(Write(label_x))
        self.wait(0.5)

        # endregion

        # region Scene 3: Basis vector y
        basis_y = Vector([0, 1], color="#FB699D")

        # add label for basis_y
        label_y = MathTex(r'\mathbf{b}_{\mathbf{e_y}}', color="#FB699D", font_size=35)  
        label_y.align_to(basis_y, UR)
        label_y.shift(0.3 * LEFT)

        self.add_vector(basis_y)
        self.play(Write(label_y))

        self.wait(0.5)
        # endregion 

        # region Scene 4: Add equation 
        equation = (MathTex(r"\textrm{g}_{r} = \mathbf{B_{r}} \mathbf{g_e}",
                            font_size=35)
                            .to_edge(UR))
        equation.add_background_rectangle()

        # Display the initial equation
        show_equation = [Create(equation)]
        self.play(*show_equation)
        self.wait(0.5)
        self.moving_mobjects = []

        # endregion    

        # region Scene 5: Show nonsquare matrix text
        # add basis vectors of 
        nonsquare_text = (MathTex(r"\mathbf{B_{r}} = \begin{bmatrix} 1  &  2 \end{bmatrix}", 
                       font_size=35)
                       .to_edge(UL))

        # # Display the initial orthonormal_text
        show_orthonormal_text = [Create(nonsquare_text)]
        self.play(*show_orthonormal_text)
        self.wait(0.5)
        self.moving_mobjects = []

        # change vector color
        vec_ge.set_color('#f5dc0e')

        # endregion
        
        # region Scene 6: Apply matrix and transformed vector label
        # non-standard basis
        self.moving_mobjects = []
        nonsquare_basis = np.array([[1,2], 
                                    [0,0]])
        
        # apply transformation 
        self.apply_matrix(nonsquare_basis)


        # show label of vector g_n
        vec_gn_label = MathTex(r'\textrm{g}_{r} = \begin{bmatrix} 4 \end{bmatrix}', 
                               color="#f5dc0e", 
                               font_size=35)
        
        vec_gn_label.add_background_rectangle()
        vec_gn_label.move_to([5.25, 0.5, 0])  
        
        show_vec_gn_label = [Write(vec_gn_label)]
        self.play(*show_vec_gn_label)
        self.wait(0.75)

        # endregion 
        
        # region 7: Change basis x label
        nonsquare_text_labels_x = (MathTex(r"\mathbf{B_{r}} = \begin{bmatrix} \overbrace{1}^{\textrm{b}_{r_x}}" + 
                                         r"&  2 \end{bmatrix}}", font_size=35).to_edge(UL))
        
        # add custom colouring
        # self.add(index_labels(text[0]))  # help determine what index values to colour
        nonsquare_text_labels_x[0][4:7].set_color("#4EF716")
        nonsquare_text_labels_x.add_background_rectangle()

        
        # label for first basis vector of orthonormal 
        nonsquare_label_x = MathTex(r'\textrm{b}_{r_x}', color="#4EF716", font_size=35)  
        nonsquare_label_x.add_background_rectangle()
        nonsquare_label_x.align_to(basis_x, UR)
        nonsquare_label_x.shift(0.4 * DOWN)


        label_matrix_vgroup = VGroup(*[nonsquare_text, nonsquare_text_labels_x])


        self.play(ReplacementTransform(label_matrix_vgroup[0], label_matrix_vgroup[1]),
                  Transform(label_x, nonsquare_label_x))

        self.wait(0.5)
        self.moving_mobjects = []  

        # endregion 

        # region 8: Change basis y label
        nonsquare_text_labels_y = (MathTex(r"\mathbf{B_{r}} = \begin{bmatrix} \overbrace{1}^{\textrm{b}_{r_x}}" + 
                                         r"& \overbrace{2}^{\textrm{b}_{r_y}} \end{bmatrix}}", font_size=35).to_edge(UL))
        
        # add custom colouring
        # self.add(index_labels(text[0]))  # help determine what index values to colour
        nonsquare_text_labels_y[0][4:7].set_color("#4EF716")
        nonsquare_text_labels_y[0][12:15].set_color("#FB699D")
        nonsquare_text_labels_y.add_background_rectangle()
        label_matrix_vgroup = VGroup(*[label_matrix_vgroup[1], nonsquare_text_labels_y])

        # label for first basis vector of orthonormal 
        nonsquare_label_y = MathTex(r'\textrm{b}_{r_y}', color="#FB699D", font_size=35)  
        nonsquare_label_y.add_background_rectangle()
        nonsquare_label_y.align_to(basis_y, UR)
        nonsquare_label_y.shift(0.47 * DOWN + 0.4 * RIGHT)


        self.play(ReplacementTransform(label_matrix_vgroup[0], label_matrix_vgroup[1]),              
                  Transform(label_y, nonsquare_label_y))
        
        self.wait(0.5)
        self.moving_mobjects = []  

        self.wait(2)

        # endregion 


class eigenvectorMatrix(LinearTransformationScene):

    def __init__(self, **kwargs):
        
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False, 
            show_coordinates=True,  # include coordinate labels
            leave_ghost_vectors=False,
            include_foreground_plane=True,  # includes grid lines 
            include_background_plane=True,  # includes numbers 
            *kwargs
        )

    def construct(self):

        # region Scene 1: Show eigenvector lines 


        # region Scene 1: Eigenvectors and their spans in standard basis 
        vec_eig1 = Vector([np.sqrt(2)/2, np.sqrt(2)/2], color='#d170c7')
        vec_eig2 = Vector([1, 0], color='#4EF716')

        # add eigenvector line for v1
        v_end_eig1 = vec_eig1.get_end()
        line_eig1 = Line(-6*v_end_eig1, 6*v_end_eig1, color = "#58edbb")

        # animation 
        self.play(Create(line_eig1))

        self.wait(0.75)

        # add eigenvector line for v2
        v_end_eig2 = vec_eig2.get_end()
        line_eig2 = Line(-8*v_end_eig2, 8*v_end_eig2, color = "#58edbb")
        self.play(Create(line_eig2))

        self.wait(0.75)


        # add eigenvectors
        vec_eig1_label = MathTex(r'\mathbf{v_1} = \begin{bmatrix} \frac{\sqrt{2}}{2} \\ \frac{\sqrt{2}}{2} \end{bmatrix}, \lambda_1 = 2', 
                               color="#d170c7", 
                               font_size=35)
        
        vec_eig2_label = MathTex(r'\mathbf{v_2} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \lambda_2 = 1', 
                       color="#4EF716", 
                       font_size=35)

        # positions 
        vec_eig1_label.add_background_rectangle()
        vec_eig1_label.move_to([5.25, 2, 0]) 

        vec_eig2_label.add_background_rectangle()
        vec_eig2_label.move_to([5.25, 0.75, 0])  
 
        # show eigenvectors and their labels
        self.add_vector(vec_eig1)
        show_vec_eig1_label = [Write(vec_eig1_label)]
        self.play(*show_vec_eig1_label)
        self.wait(0.75)

        self.add_vector(vec_eig2)
        show_vec_eig2_label = [Write(vec_eig2_label)]
        self.play(*show_vec_eig2_label)
        self.wait(0.75)

        
        # endregion 

        # region Scene 2: Basis vector x
        # basis_x = Vector([1, 0], color="#4EF716")

        # add label for basis_x 
        # abel_x = MathTex(r'\mathbf{b}_{\mathbf{e_x}}', color="#4EF716", font_size=35)  
        # abel_x.align_to(vec_eig2, UL)
        # abel_x.shift(0.25 * DOWN + 0.45 * RIGHT)

        # elf.wait(0.5)
        # elf.play(Write(label_x))
        # elf.wait(0.5)

        # endregion

        # region Scene 3: Basis vector y
        # basis_y = Vector([0, 1], color="#FB699D")

        # # add label for basis_y
        # label_y = MathTex(r'\mathbf{b}_{\mathbf{e_y}}', color="#FB699D", font_size=35)  
        # label_y.align_to(basis_y, UR)
        # label_y.shift(0.3 * LEFT)

        # self.add_vector(basis_y)
        # self.play(Write(label_y))

        # self.wait(0.5)
        # endregion 

        # region Scene 5: Show eigenvector matrix text
        # add basis vectors of 
        nonsquare_text = (MathTex(r"\mathbf{A} = \begin{bmatrix} 1  &  1 \\\\ 0 & 2 \end{bmatrix}", 
                       font_size=35)
                       .to_edge(UL))
        

        # # Display the initial orthonormal_text
        show_orthonormal_text = [Create(nonsquare_text)]
        self.play(*show_orthonormal_text)
        self.wait(0.5)
        self.moving_mobjects = []

        

        # endregion

        # region Scene 4: Add equation 
        equation = (MathTex(r"\mathbf{Av}= \lambda \mathbf{v}",
                            font_size=35))
        
        equation.add_background_rectangle()
        equation.move_to([2, -1, 0])

        # Display the initial equation
        show_equation = [Create(equation)]
        self.play(*show_equation)
        self.wait(0.5)
        self.moving_mobjects = []

        # endregion    
        
        # region Scene 6: Apply matrix and  show eigenvectorstransformed vector label
        # non-standard basis
        self.moving_mobjects = []
        eigen_matrix = np.array([[1,1], 
                                 [0,2]])
        
        # apply transformation 
        self.apply_matrix(eigen_matrix)
        vec_eig1.set_color('#f5dc0e')

        # show computation of eigenvector 1
        eig_vec_1 = MathTex(r'\lambda_1 \mathbf{v_1} = \begin{bmatrix} \sqrt{2} \\ \sqrt{2} \end{bmatrix} ', 
                               color="#f5dc0e", 
                               font_size=35)
        
        eig_vec_1.add_background_rectangle()
        eig_vec_1.move_to([5, -1, 0])
        
        show_eig_vec_1_label = [Write(eig_vec_1)]
        self.play(*show_eig_vec_1_label)
        self.wait(0.75)


        # show computation of eigenvector 1
        vec_eig2.set_color('#FF33F6')

        eig_vec_2 = MathTex(r'\lambda_2 \mathbf{v_2} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} ', 
                               color="#FF33F6", 
                               font_size=35)
        
        eig_vec_2.add_background_rectangle()
        eig_vec_2.move_to([5, -2, 0])
        
        show_eig_vec_2_label = [Write(eig_vec_2)]
        self.play(*show_eig_vec_2_label)
        self.wait(0.75)


        # endregion 
        


import cv2
from PIL import Image, ImageOps
from dataclasses import dataclass

@dataclass
class VideoStatus:
    time: float = 0
    videoObject: cv2.VideoCapture = None
    def __deepcopy__(self, memo):
        return self

class VideoMobject(ImageMobject):
    '''
    Following a discussion on Discord about animated GIF images.
    Modified for videos
    Parameters
    ----------
    filename
        the filename of the video file
    imageops
        (optional) possibility to include a PIL.ImageOps operation, e.g.
        PIL.ImageOps.mirror
    speed
        (optional) speed-up/slow-down the playback
    loop
        (optional) replay the video from the start in an endless loop
    https://discord.com/channels/581738731934056449/1126245755607339250/1126245755607339250
    2023-07-06 Uwe Zimmermann & Abulafia
    2024-03-09 Uwe Zimmermann
    '''
    def __init__(self, filename=None, imageops=None, speed=1.0, loop=False, **kwargs):
        self.filename = filename
        self.imageops = imageops
        self.speed    = speed
        self.loop     = loop
        self._id = id(self)
        self.status = VideoStatus()
        self.status.videoObject = cv2.VideoCapture(filename)

        self.status.videoObject.set(cv2.CAP_PROP_POS_FRAMES, 1)
        ret, frame = self.status.videoObject.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)            
            img = Image.fromarray(frame)

            if imageops != None:
                img = imageops(img)
        else:
            img = Image.fromarray(np.uint8([[63, 0, 0, 0],
                                        [0, 127, 0, 0],
                                        [0, 0, 191, 0],
                                        [0, 0, 0, 255]
                                        ]))
        super().__init__(img, **kwargs)
        if ret:
            self.add_updater(self.videoUpdater)

    def videoUpdater(self, mobj, dt):
        if dt == 0:
            return
        
        status = self.status
        status.time += 1000*dt*mobj.speed

        self.status.videoObject.set(cv2.CAP_PROP_POS_MSEC, status.time)

        ret, frame = self.status.videoObject.read()

        if (ret == False) and self.loop:
            status.time = 0
            self.status.videoObject.set(cv2.CAP_PROP_POS_MSEC, status.time)
            ret, frame = self.status.videoObject.read()

        if ret:
            frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)         
            img = Image.fromarray(frame)

            if mobj.imageops != None:
                img = mobj.imageops(img)
            mobj.pixel_array = change_to_rgba_array(
                np.asarray(img), mobj.pixel_array_dtype
            )


class svd_anim_proof(Scene):
    def construct(self):
       
        # region 0) Add video 
        video1 = VideoMobject(
            filename=r"/Users/sebastiansciarra/Desktop/Projects/HugoWebsite/content/technical_content/svd/media/videos/anim/480p15/svd.mp4",
            speed=1.0
        ).scale_to_fit_width(10).center().shift(0.55 * UP)

        v1 = Group(video1, SurroundingRectangle(video1, color='#33c304'))
        self.add(v1)

        # endregion 

        # region 1) Add matrix 
        matrix = MathTex(
            r"\mathbf{A} = \begin{bmatrix} 2 & 1 & 1 \\  0 & 2 & 3 \end{bmatrix}", 
            font_size=28)
        
        # Move the expression to a fixed location
        matrix.center().shift(3 * DOWN)
        self.add(matrix)

        self.wait(10)

        # endregion 

        # region 2) Add basis vector x 
        matrix_label_x = (MathTex(
            r"\begin{array}{ccc}\begin{matrix}" + 
            r"\hspace{0.75cm} \mathbf{b_{x}} &  \phantom{\mathbf{b_{y}}} " + 
            r"& \phantom{\mathbf{b_z}} \end{matrix} \\ " +
            r"\mathbf{A} = \begin{bmatrix} 2 & \hspace{0.1cm} 1 \hspace{0.1cm}" + 
            r"& \hspace{0.1cm} 1 \\ \hspace{0.04cm}  0 \hspace{0.1cm} & \hspace{0.1cm} 2 \hspace{0.1cm}" + 
            r"\hspace{0.1cm} & 3 \end{bmatrix} \end{array}", 
            font_size=28))
        
        matrix_label_x.center().shift(3 * DOWN)
        # add custom colouring
        # self.add(index_labels(matrix_label_x[0]))  # help determine what index values to colour
        matrix_label_x[0][0:2].set_color("#4EF716")
        
        self.play(Transform(matrix, matrix_label_x))
        self.wait(2)
        self.moving_mobjects = []  
        # endregion 

        # region 3) Add basis vector y
        matrix_label_y = (MathTex(
            r"\begin{array}{ccc}\begin{matrix}" + 
            r"\hspace{0.75cm} \mathbf{b_{x}} &  \mathbf{b_{y}} " + 
            r"& \phantom{\mathbf{b_z}}  \end{matrix} \\ " +
            r"\mathbf{A} = \begin{bmatrix} 2 & \hspace{0.1cm} 1 \hspace{0.1cm}" + 
            r"& \hspace{0.1cm} 1 \\ \hspace{0.04cm}  0 \hspace{0.1cm} & \hspace{0.1cm} 2 \hspace{0.1cm}" + 
            r"\hspace{0.1cm} & 3 \end{bmatrix} \end{array}", 
            font_size=28))
        
        matrix_label_y.center().shift(3 * DOWN)
        # add custom colouring
        # self.add(index_labels(matrix_label_y[0]))  # help determine what index values to colour
        matrix_label_y[0][0:2].set_color("#4EF716")
        matrix_label_y[0][2:4].set_color("#FB699D")

        self.play(Transform(matrix_label_x, matrix_label_y))

        self.wait(2)
        self.moving_mobjects = []  
        # endregion 

        # region 4) Add basis vector y
        matrix_label_z = (MathTex(
            r"\begin{array}{ccc}\begin{matrix}" + 
            r"\hspace{0.75cm} \mathbf{b_{x}} &  \mathbf{b_{y}} " + 
            r"&  \mathbf{b_{z}} \end{matrix} \\ " +
            r"\mathbf{A} = \begin{bmatrix} 2 & \hspace{0.1cm} 1 \hspace{0.1cm}" + 
            r"& \hspace{0.1cm} 1 \\ \hspace{0.04cm}  0 \hspace{0.1cm} & \hspace{0.1cm} 2 \hspace{0.1cm}" + 
            r"\hspace{0.1cm} & 3 \end{bmatrix} \end{array}", 
            font_size=28))
        
        matrix_label_z.center().shift(3 * DOWN)
        # add custom colouring
        # self.add(index_labels(matrix_label_z[0]))  # help determine what index values to colour
        matrix_label_z[0][0:2].set_color("#4EF716")
        matrix_label_z[0][2:4].set_color("#FB699D")
        matrix_label_z[0][4:6].set_color("#52aafa")


        self.play(Transform(matrix_label_x, matrix_label_z))

        self.wait(2)
        self.moving_mobjects = []  
        self.remove(matrix, matrix_label_y, matrix_label_x)
        # endregion 
        
        # region 5) Move equation and video 
        mat_copy = matrix_label_z.copy()
        mat_copy.shift(3*LEFT)

        self.play(Transform(v1, v1.copy().scale_to_fit_width(2).to_corner(UL + 0.15*DOWN)), 
                  Transform(mobject=matrix_label_z, 
                            target_mobject=mat_copy, 
                            replace_mobject_with_target_in_scene=True)) 
        self.remove(matrix_label_z)
        self.wait(2)

        # endregion 

        # region 6) Expand equation 
        matrix_svd_label = (MathTex(
                    r"\begin{array}{ccc}\begin{matrix}" + 
                    r"\hspace{0.75cm} \mathbf{b_{x}} &  \mathbf{b_{y}} " + 
                    r"&  \mathbf{b_{z}} \end{matrix} \\ " +
                    r"\mathbf{A} = \begin{bmatrix} 2 & \hspace{0.1cm} 1 \hspace{0.1cm}" + 
                    r"& \hspace{0.1cm} 1 \\ \hspace{0.04cm}  0 \hspace{0.1cm} & \hspace{0.1cm} 2 \hspace{0.1cm}" + 
                    r"\hspace{0.1cm} & 3 \end{bmatrix} \end{array}" + 
                    r"= \mathbf{U \Sigma V^\top}", 
                    font_size=28))
        
        matrix_svd_label.center().shift(3 * DOWN + 3*LEFT)
        # add custom colouring
        # self.add(index_labels(matrix_svd_label[0]))  # help determine what index values to colour
        matrix_svd_label[0][0:2].set_color("#4EF716")
        matrix_svd_label[0][2:4].set_color("#FB699D")
        matrix_svd_label[0][4:6].set_color("#52aafa")

        self.play(Transform(mobject=mat_copy, 
                            target_mobject=matrix_svd_label, 
                            replace_mobject_with_target_in_scene=True)) 

        self.wait(1)
        # endregion 

        # region 7) Add V^t to equation 
        matrix_svd_v_t = (MathTex(
                    r"\begin{array}{ccc}\begin{matrix}" + 
                    r"\hspace{0.75cm} \mathbf{b_{x}} &  \mathbf{b_{y}} " + 
                    r"&  \mathbf{b_{z}} \end{matrix} \\ " +
                    r"\mathbf{A} = \begin{bmatrix} 2 & \hspace{0.1cm} 1 \hspace{0.1cm}" + 
                    r"& \hspace{0.1cm} 1 \\ \hspace{0.04cm}  0 \hspace{0.1cm} & \hspace{0.1cm} 2 \hspace{0.1cm}" + 
                    r"\hspace{0.1cm} & 3 \end{bmatrix} \end{array}" + 
                    r"= \mathbf{U \Sigma V^\top}" + 
                    r"= \mathbf{U \Sigma} \underbrace{\begin{bmatrix} -0.23 & -0.57 & -0.79 \\" + 
                    r"-0.96 & 0.01 & 0.27 \\ 0.14 & -0.82 & 0.55 \end{bmatrix}}_{\text{Rotation}}",
                    font_size=28))
        
        matrix_svd_v_t.center().shift(3 * DOWN + 1.5*LEFT)
        # add custom colouring
        # self.add(index_labels(matrix_svd_v_t[0]))  # help determine what index values to colour
        matrix_svd_v_t[0][0:2].set_color("#4EF716")
        matrix_svd_v_t[0][2:4].set_color("#FB699D")
        matrix_svd_v_t[0][4:6].set_color("#52aafa")
    
        self.play(Transform(mobject=matrix_svd_label, 
                            target_mobject=matrix_svd_v_t, 
                            replace_mobject_with_target_in_scene=True)) 

        self.wait(2)

        # endregion 
        
        # region 8) Add video + wiggle V^t + add video 
        video1 = VideoMobject(
            filename=r"/Users/sebastiansciarra/Desktop/Projects/HugoWebsite/content/technical_content/svd/media/videos/anim/480p15/svd_piecemeal.mp4",
            speed=1.0
        ).scale_to_fit_width(9.75).center().shift(0.72 * UP  + 0.75*RIGHT)

        v2 = Group(video1, SurroundingRectangle(video1, color='#33c304'))
        self.add(v2)

        self.play(Wiggle(mobject=matrix_svd_v_t[0][19:21]), 
                  Wiggle(mobject=matrix_svd_v_t[0][24:69]))
        self.wait(2.5)
        # endregion 

        # region 9) Add sigma + play transformation 
        matrix_svd_sigma = (MathTex(
            r"\begin{array}{ccc}\begin{matrix}" + 
            r"\hspace{0.75cm} \mathbf{b_{x}} &  \mathbf{b_{y}} " + 
            r"&  \mathbf{b_{z}} \end{matrix} \\ " +
            r"\mathbf{A} = \begin{bmatrix} 2 & \hspace{0.1cm} 1 \hspace{0.1cm}" + 
            r"& \hspace{0.1cm} 1 \\ \hspace{0.04cm}  0 \hspace{0.1cm} & \hspace{0.1cm} 2 \hspace{0.1cm}" + 
            r"\hspace{0.1cm} & 3 \end{bmatrix} \end{array}" + 
            r"= \mathbf{U \Sigma V^\top}" + 
            r"= \mathbf{U} \underbrace{\begin{bmatrix} 3.95 & 0 & 0 \\" + 
            r"0 & 1.84 & 0 \end{bmatrix}}_{\text{Stretch + dim. change}}" + 
            r"\underbrace{\begin{bmatrix} -0.23 & -0.57 & -0.79 \\" + 
            r"-0.96 & 0.01 & 0.27 \\ 0.14 & -0.82 & 0.55 \end{bmatrix}}_{\text{Rotation}}",
            font_size=28))
        
        matrix_svd_sigma.center().shift(3 * DOWN + 1*LEFT)
        # add custom colouring
        # self.add(index_labels(matrix_svd_sigma[0]))  # help determine what index values to colour
        matrix_svd_sigma[0][0:2].set_color("#4EF716")
        matrix_svd_sigma[0][2:4].set_color("#FB699D")
        matrix_svd_sigma[0][4:6].set_color("#52aafa")
    
        self.play(Transform(mobject=matrix_svd_v_t, 
                            target_mobject=matrix_svd_sigma, 
                            replace_mobject_with_target_in_scene=True)) 

        self.wait(1)

        self.play(Wiggle(mobject=matrix_svd_sigma[0][18:19]), 
                  Wiggle(mobject=matrix_svd_sigma[0][23:37]))
        
        self.wait(7)
        # endregion 

        # region 10) Add U + play transformation 
        matrix_svd_u = (MathTex(
            r"\begin{array}{ccc}\begin{matrix}" + 
            r"\hspace{0.75cm} \mathbf{b_{x}} &  \mathbf{b_{y}} " + 
            r"&  \mathbf{b_{z}} \end{matrix} \\ " +
            r"\mathbf{A} = \begin{bmatrix} 2 & \hspace{0.1cm} 1 \hspace{0.1cm}" + 
            r"& \hspace{0.1cm} 1 \\ \hspace{0.04cm}  0 \hspace{0.1cm} & \hspace{0.1cm} 2 \hspace{0.1cm}" + 
            r"\hspace{0.1cm} & 3 \end{bmatrix} \end{array}" + 
            r"= \mathbf{U \Sigma V^\top}" + 
            r"= \underbrace{\begin{bmatrix} -0.46 & -0.89 & \\ -0.89 & 0.46 \end{bmatrix}}_{\text{Rotation}}" + 
            r" \underbrace{\begin{bmatrix} 3.95 & 0 & 0 \\" + 
            r"0 & 1.84 & 0 \end{bmatrix}}_{\text{Stretch + dim. change}}" + 
            r"\underbrace{\begin{bmatrix} -0.23 & -0.57 & -0.79 \\" + 
            r"-0.96 & 0.01 & 0.27 \\ 0.14 & -0.82 & 0.55 \end{bmatrix}}_{\text{Rotation}}",
            font_size=28))
        
        matrix_svd_u.center().shift(3 * DOWN + 0.5*LEFT)
        # add custom colouring
        # self.add(index_labels(matrix_svd_u[0]))  # help determine what index values to colour
        matrix_svd_u[0][0:2].set_color("#4EF716")
        matrix_svd_u[0][2:4].set_color("#FB699D")
        matrix_svd_u[0][4:6].set_color("#52aafa")
    
        self.play(Transform(mobject=matrix_svd_sigma, 
                            target_mobject=matrix_svd_u, 
                            replace_mobject_with_target_in_scene=True)) 

        self.wait(1)

        self.play(Wiggle(mobject=matrix_svd_u[0][17:18]), 
                  Wiggle(mobject=matrix_svd_u[0][22:42]))
        
        self.wait(2)
        # endregion 

        # region 11) Wiggle each basis vector 
        self.play(Wiggle(mobject=matrix_svd_u[0][0:2]))
        self.wait(0.5)

        self.play(Wiggle(mobject=matrix_svd_u[0][2:4]))
        self.wait(0.5)

        self.play(Wiggle(mobject=matrix_svd_u[0][4:6]))
        self.wait(1.25)
        # endregion 

        # region 12) Bring videos side by side 
        self.play(Transform(v1, v1.copy().scale_to_fit_width(6).to_corner(UL + 0.15*DOWN)),
                  Transform(v2, v2.copy().scale_to_fit_width(6).to_corner(UR + 0.15*DOWN)))

        self.wait(1.5)


class svd(ThreeDScene):
    def construct(self):
        # Set up axes
        axes = ThreeDAxes(x_range=[-7, 7, 1], 
                          x_length=14)
        
        self.set_camera_orientation(phi= 75* DEGREES, theta=-45 * DEGREES)

        # region 1) Create basis vectors and apply transformation 

        # Define a vector in 3D
        
        basis_x = Vector([1, 0, 0], color='#4EF716')
        basis_y = Vector([0, 1, 0], color='#FB699D')
        basis_z = Vector([0, 0, 1], color='#52aafa')

        basis_vectors = VGroup(basis_x, basis_y, basis_z)

        # Define a 3x3 transformation matrix for scaling and rotating
        transformation_matrix = [[2, 1, 1],  # Scaling along x-axis
                                 [0, 2, 3],  # No change along y-axis
                                 [0, 0, 0]]  # No change along z-axis
        
        # Display the initial vector
        self.play(Create(axes), Create(basis_x), Create(basis_y), Create(basis_z))
        self.wait(1)

        # Apply the transformation matrix
        self.play(ApplyMatrix(transformation_matrix, basis_vectors))
        self.wait(1.5)

        # remove and replace basis vectors so they scale appropriately 
        anim_remove_basis = FadeOut(basis_vectors)

        new_basis_x = Vector(direction=[2, 0, 0], color='#4EF716')
        new_basis_y = Vector(direction=[1, 2, 0], color='#FB699D')
        new_basis_z = Vector(direction=[1, 3, 0], color='#52aafa')
        new_basis = VGroup(new_basis_x, new_basis_y, new_basis_z)

        create_basis = Create(new_basis)

        self.move_camera(phi=0 * DEGREES, theta= -90 * DEGREES, 
                         added_anims=[anim_remove_basis, create_basis], 
                         run_time=2)
        self.wait(1.5)
        # endregion 

        # region 2) Show basis vector x 
        basis_x_text = (MathTex(
            r"\mathbf{b}_x = \begin{bmatrix} 2 \\ 0 \end{bmatrix}", 
            font_size=34, color='#4EF716'))
                     
        basis_x_text.move_to(new_basis_x.get_end() + np.array([0.2, -0.6, 0]))  # Slightly above the end of the vector
        
        # Add the vector and label to the scene
        self.add(new_basis_x, basis_x_text)
        
        self.wait(2)

        # endregion 

        # region 3) Show basis vector y 
        basis_y_text = (MathTex(
            r"\mathbf{b}_y = \begin{bmatrix} 1 \\ 2 \end{bmatrix}", 
            font_size=34, color='#FB699D'))
                     
        basis_y_text.move_to(new_basis_y.get_end() + np.array([0.7, -0.5, 0]))  # Slightly above the end of the vector
        
        # Add the vector and label to the scene
        self.add(new_basis_y, basis_y_text)
        
        self.wait(3)

        # endregion 

        # region 4) Show basis vector z
        basis_z_text = (MathTex(
            r"\mathbf{b}_z = \begin{bmatrix} 1 \\ 3 \end{bmatrix}", 
            font_size=34, color='#52aafa'))
                     
        basis_z_text.move_to(new_basis_z.get_end() + np.array([0.75, 0, 0]))  # Slightly above the end of the vector
        
        # Add the vector and label to the scene
        self.add(new_basis_z, basis_z_text)
        
        self.wait(2)

        # endregion 

class svd_piecemeal(ThreeDScene):
    def construct(self): 
        # Set up axes
        axes = ThreeDAxes(x_range=[-7, 7, 1], 
                          x_length=14)
        
        self.set_camera_orientation(phi= 75* DEGREES, theta=-135 * DEGREES)

        # region 1) Create basis vectors and apply Vt 

        # Define a vector in 3D
        
        basis_x = Vector([1, 0, 0], color='#4EF716')
        basis_y = Vector([0, 1, 0], color='#FB699D')
        basis_z = Vector([0, 0, 1], color='#52aafa')

        basis_vectors = VGroup(basis_x, basis_y, basis_z)

        # Define a 3x3 transformation matrix for scaling and rotating
        V_t = [[-0.23382221, -0.56600304, -0.79054901],
               [-0.96252753,  0.01988155,  0.2704542 ],
               [ 0.13736056, -0.82416338,  0.54944226]]
        
        # Display the initial vector
        self.play(Create(axes), Create(basis_x), Create(basis_y), Create(basis_z))
        self.wait(1)

        # Apply the transformation matrix
        self.play(ApplyMatrix(V_t, basis_vectors))
        self.wait(2.5)
        # endregion 

        # region 2) Apply sigma transformation 
        sigma = [[3.95009846, 0, 0], 
                 [0, 1.84301986, 0], 
                 [0, 0, 0]]
        
        self.play(ApplyMatrix(sigma, basis_vectors))
        self.wait(1.5)


        new_basis_x = Vector(direction=[-0.92362075, -1.77395735, 0], color='#4EF716')
        new_basis_y = Vector(direction=[-2.23576774, 0.03664209, 0], color='#FB699D')
        new_basis_z = Vector(direction=[-3.12274643, 0.49845246, 0], color='#52aafa')
        new_basis = VGroup(new_basis_x, new_basis_y, new_basis_z)

        create_basis = Create(new_basis)

        # remove and replace basis vectors so they scale appropriately 
        anim_remove_basis = FadeOut(basis_vectors)

        self.move_camera(phi=0 * DEGREES, theta= -90 * DEGREES, 
                         added_anims=[anim_remove_basis, create_basis], 
                         run_time=2)

        self.wait(4.5)

        # endregion  

        # region 3) Apply U
        U = [[-0.46181038, -0.88697868],
             [-0.88697868,  0.46181038]]
        
        self.play(ApplyMatrix(U, new_basis))
        self.wait(1.5)

        # endregion 


        # region 4) Show basis vector x 
        basis_x_text = (MathTex(
            r"\mathbf{b}_x = \begin{bmatrix} 2 \\ 0 \end{bmatrix}", 
            font_size=34, color='#4EF716'))
                     
        basis_x_text.move_to(new_basis_x.get_end() + np.array([0.2, -0.6, 0]))  # Slightly above the end of the vector
        
        # Add the vector and label to the scene
        self.add(new_basis_x, basis_x_text)
        
        self.wait(2)

        # endregion 

        # region 5) Show basis vector y 
        basis_y_text = (MathTex(
            r"\mathbf{b}_y = \begin{bmatrix} 1 \\ 2 \end{bmatrix}", 
            font_size=34, color='#FB699D'))
                     
        basis_y_text.move_to(new_basis_y.get_end() + np.array([0.7, -0.5, 0]))  # Slightly above the end of the vector
        
        # Add the vector and label to the scene
        self.add(new_basis_y, basis_y_text)
        
        self.wait(3)

        # endregion 

        # region 6) Show basis vector z
        basis_z_text = (MathTex(
            r"\mathbf{b}_z = \begin{bmatrix} 1 \\ 3 \end{bmatrix}", 
            font_size=34, color='#52aafa'))
                     
        basis_z_text.move_to(new_basis_z.get_end() + np.array([0.75, 0, 0]))  # Slightly above the end of the vector
        
        # Add the vector and label to the scene
        self.add(new_basis_z, basis_z_text)
        
        self.wait(2)

        # endregion 






class VectorProjection(LinearTransformationScene): 
  
  def __init__(self, **kwargs):
        
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False, 
            show_coordinates=True,  # include coordinate labels
            leave_ghost_vectors=False,
            include_foreground_plane=True,  # includes grid lines 
            include_background_plane=True,  # includes numbers 
            *kwargs
        )

  def construct(self): 
    plane = NumberPlane()
    vecu = [-2, 3, 0]
    vecv = [np.sqrt(2)/2, np.sqrt(2)/2, 0]

    arrowU = Vector(vecu, buff=0, color = YELLOW)
    # add label for basis_i
    label_u = MathTex(r'\hat{j}', color="#33c304", font_size=40)  
    label_u.align_to(arrowU, UL)
    label_u.shift(0.3 * LEFT)

    # vectorU = Vector()
    arrowU1 = Vector(vecu, buff=0, color = YELLOW)
    arrowV = Vector(vecv, buff = 0, color = BLUE)

    # Compute Projection of U onto V
    numerator = np.dot(vecu, vecv)
    demoninator = np.linalg.norm(vecv)**2 
    scalar = numerator / demoninator
    vecProjUtoV = scalar * np.array(vecv) 
    ArrowProjection = Vector(vecProjUtoV, buff=0, color = PINK) 

    # Compute Line orthogonal to V
    line = Line(vecu, vecProjUtoV, buff=0, color = GREY)
    line2 = Line(ORIGIN, vecv, buff = 0)
    Rangle = RightAngle(line2, line, length=0.4, color = GREY, 
                        quadrant = (-1, -1)) 

    # Animation
    # self.play(Create(plane), run_time = 1)
    self.play(GrowArrow(arrowU), GrowArrow(arrowV))
    self.add(arrowU1)
    self.wait()
    self.play(Create(line))
    self.play(Create(Rangle))
    self.wait()
    self.play(Transform(arrowU1, ArrowProjection))
    self.play(FadeOut(line), FadeOut(Rangle))
    self.wait(2)


class VectorProjection1(Scene): 
  def construct(self): 
    plane = NumberPlane()
    coord_vecU = [-2, 3, 0]
    coord_vecV = [1, 2, 0]

    vecU = Vector(coord_vecU, buff=0, color = YELLOW)
    # vectorU = Vector()
   #  arrowU1 = Vector(ORIGIN, coord_vecU, buff=0, color = YELLOW)
    vecV = Vector(coord_vecV, buff = 0, color = BLUE)

    # Compute Projection of U onto V
    numerator = np.dot(coord_vecU, coord_vecV)
    demoninator = np.linalg.norm(coord_vecV)**2 
    scalar = numerator / demoninator
    vecProjUtoV = scalar * np.array(coord_vecV) 
    ArrowProjection = Arrow(ORIGIN, vecProjUtoV, buff=0, color = PINK) 

    # Compute Line orthogonal to V
    line = Line(coord_vecU, vecProjUtoV, buff=0, color = GREY)
    line2 = Line(ORIGIN, coord_vecV, buff = 0)
    Rangle = RightAngle(line2, line, length=0.4, color = GREY, 
                        quadrant = (-1, -1)) 

    # Animation
    self.play(Create(plane), run_time = 1)
    self.play(GrowArrow(vecU), GrowArrow(vecV))
    self.add(arrowU1)
    self.wait()
    self.play(Create(line))
    self.play(Create(Rangle))
    self.wait()
    self.play(Transform(arrowU1, ArrowProjection))
    self.play(FadeOut(line), FadeOut(Rangle))
    self.wait(2)