class Field:
    """
        Short class for electron/ion density
    """
    def __init__(self, data):
        import happi
        from numpy import pi

        self.value  = happi.Open(data)
        self.L0     = 2 * pi
        self.resx   = 50

    @property
    def eons(self):
        return self.value.Field(0, '-Rho_electron')

    @property
    def ions(self):
        return self.value.Field(0, 'Rho_ion')

    def get_e_field(self, time, axis='x'):
        if axis in ['x', 'y', 'z']:
            return self.value.Field(0, f'E{axis}').getData(timestep=time)[0]

    def get_b_field(self, time, axis='x'):
        if axis in ['x', 'y', 'z']:
            return self.value.Field(0, f'B{axis}').getData(timestep=time)[0]

    @property
    def x(self):
        return self.eons.getAxis(axis='x')/self.L0

    @property
    def y(self):
        return self.eons.getAxis(axis='y')/self.L0

    @staticmethod
    def plot(x, y, m_dict):
        import pylab as pl

        fig = pl.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        ax.plot(x, y, color='red', linewidth=0.2, label=m_dict['label'])
        ax.set_xlim(m_dict['xmin'], m_dict['xmax'])
        ax.set_ylim(m_dict['ymin'], m_dict['ymax'])
        ax.set_xlabel(m_dict['xlabel'], fontsize=7)
        ax.set_ylabel(m_dict['ylabel'], fontsize=7)
        ax.legend()

        ax.minorticks_on()
        ax.grid(which='major', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=7)

        fig.savefig(m_dict['name'], bbox_inches='tight', pad_inches=0.1, dpi=300)
        pl.close(fig)

    def plot_ne(self, time):
        print('plot the ne electron density distribution')

        ne     = self.eons.getData(timestep=time)[0]
        x      = self.x

        m_dict = {'label'   :'ne',
                  'xmin'    : None,
                  'xmax'    : None,
                  'xlabel'  : r'$\mathrm{x \ [\lambda_0]}$',
                  'ymin'    : None,
                  'ymax'    : None,
                  'ylabel'  : r'$\mathrm{ne \ [n_c]}$',
                  'name'    : f'ne_{time}.png'}

        self.plot(x, ne, m_dict)

    def plot_ni(self, time):
        print('plot the ni electron density distribution')

        ni     = self.ions.getData(timestep=time)[0]
        x      = self.x

        m_dict = {'label'   :'ni',
                  'xmin'    : None,
                  'xmax'    : None,
                  'xlabel'  : r'$\mathrm{x \ [\lambda_0]}$',
                  'ymin'    : None,
                  'ymax'    : None,
                  'ylabel'  : r'$\mathrm{ni \ [n_c]}$',
                  'name'    : f'ni_{time}.png'}

        self.plot(x, ne, m_dict)


    def plot_ne_fft(self, time):
        from numpy import where

        print('plot the fft ne electron density spectr near nc/4')

        ne     = self.eons.getData(timestep=time)[0]
        n0     = self.eons.getData(timestep=0)[0]
        x      = self.x


        idx    = where((x>110) & (x<125))
        ne     = ne[idx]
        x      = x[idx]

        ne_fft = fft(ne-n0)
        kx     = 50*fftfreq(x.shape[0])

        m_dict = {'label'   :'ne',
                  'xmin'    : None,
                  'xmax'    : None,
                  'xlabel'  : r'$\mathrm{kx \ [\lambda_0]}$',
                  'ymin'    : None,
                  'ymax'    : None,
                  'ylabel'  : r'$\mathrm{ne \ [n_c]}$',
                  'name'    : f'ne_fft_{time}.png'}

        self.plot(kx, ne_fft, m_dict)

    def plot_e_field(self, time, axis='y'):
        from numpy import where

        print(f'plot the e{axis} field near nc/4')

        e_f    = self.get_e_field(time=time, axis=axis)
        x      = self.x

        idx    = where((x>110) & (x<125))
        e_f    = e_f[idx]
        x      = x[idx]

        m_dict = {'label'   :f'e{axis}',
                  'xmin'    : None,
                  'xmax'    : None,
                  'xlabel'  : r'$\mathrm{x \ [\lambda_0]}$',
                  'ymin'    : None,
                  'ymax'    : None,
                  'ylabel'  : f'e{axis}' + r'$\mathrm{ \ [m_e c \omega / e]}$',
                  'name'    : f'e{axis}_{time}.png'}

        self.plot(x, e_f, m_dict)

    def plot_b_field(self, time, axis='z'):
        from numpy import where

        print(f'plot the b{axis} field near nc/4')

        b_f    = self.get_b_field(time=time, axis=axis)
        x      = self.x

        idx    = where((x>110) & (x<125))
        b_f    = b_f[idx]
        x      = x[idx]

        m_dict = {'label'   :f'b{axis}',
                  'xmin'    : None,
                  'xmax'    : None,
                  'xlabel'  : r'$\mathrm{x \ [\lambda_0]}$',
                  'ymin'    : None,
                  'ymax'    : None,
                  'ylabel'  : f'b{axis}' + r'$\mathrm{ \ [m_e c \omega / e]}$',
                  'name'    : f'b{axis}_{time}.png'}

        self.plot(x, b_f, m_dict)

    def plot_sx(self, time, axis='x'):
        from numpy import where

        print(f'plot the s{axis} poynting vector')

        e_f    = lambda ax: self.get_e_field(time=time, axis=ax)
        b_f    = lambda ax: self.get_b_field(time=time, axis=ax)
        s_f    = e_f('y')*b_f('z') - e_f('z')*b_f('y')
        x      = self.x

        idx    = where((x>0) & (x<25))
        s_f    = s_f[idx]
        x      = x[idx]

        m_dict = {'label'   :f's{axis}',
                  'xmin'    : None,
                  'xmax'    : None,
                  'xlabel'  : r'$\mathrm{x \ [\lambda_0]}$',
                  'ymin'    : None,
                  'ymax'    : None,
                  'ylabel'  : f's{axis}' + r'$\mathrm{ \ [c (m_e \omega / e)^2]}$',
                  'name'    : f's{axis}_{time}.png'}

        self.plot(x, s_f, m_dict)


if __name__ == '__main__':
    f = Field('../1D_test/')
    f.plot_e_field(10000, 'y')
    f.plot_b_field(10000, 'z')
  
