class Field:
    """
        Short class for electron/ion density
    """
    def __init__(self, data):
        import happi
        from numpy import pi

        self.value = happi.Open(data)
        self.L0 = 2 * pi

    @property
    def eons(self):
        return self.value.Field(0, '-Rho_electron')

    @property
    def ions(self):
        return self.value.Field(0, 'Rho_ion')

    def get_e_field(self, time, axis='x'):
        if direction in ['x', 'y', 'z']:
            return self.value.Field(0, f'E{axis}').getData(timestep=time)

    def get_b_field(self, time, axis='x'):
        if direction in ['x', 'y', 'z']:
            return self.value.Field(0, f'B{axis}').getData(timestep=time)

    @property
    def x(self):
        return self.eons.getAxis(axis='x')/self.L0

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

        ne = self.eons.getData(timestep=time)[0]
        x = self.x

        m_dict = {'label'   :'ne',
                  'xmin'    : None,
                  'xmax'    : None,
                  'xlabel'  : r'$\mathrm{x \ [\lambda_0]}$',
                  'ymin'    : None,
                  'ymax'    : None,
                  'ylabel'  : r'$\mathrm{ne \ [n_c]}$',
                  'name'    : f'ne_{time}.png'}

        self.plot(x, ne, m_dict)


if __name__ == '__main__':
    f = Field('1D_test/')
    f.plot_ne(10000)