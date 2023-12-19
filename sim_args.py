from dataclasses import asdict
from typing_extensions import Annotated

from annotated_types import Ge, Gt
from pydantic.dataclasses import dataclass


@dataclass
class TcpRlSimArgs:
    """Definition of simulation arguments accepted by sim.cc"""

    simSeed: Annotated[int, Ge(0)] = 0
    """Seed for random generator. Default: 0"""

    envTimeStep: Annotated[float, Gt(0)] = 0.1
    """Time step interval for TcpRlTimeBased. Default: 0.1s"""

    envTimeStep: Annotated[float, Gt(0)] = 0.1
    """Time step interval for TcpRlTimeBased. Default: 0.1s"""

    nLeaf: Annotated[int, Gt(0)] = 1
    """Number of left and right side leaf nodes"""

    error_p: Annotated[float, Ge(0)] = 0
    """Packet error rate"""

    bottleneck_bandwidth: str = "2Mbps"
    """Bottleneck bandwidth"""

    bottleneck_delay: str = "0.01ms"
    """Bottleneck delay"""

    access_bandwidth: str = "10Mbps"
    """Access link bandwidth"""

    access_delay: str = "20ms"
    """Access link delay"""

    prefix_name: str = "TcpVariantsComparison"
    """Prefix of output trace file"""

    data: Annotated[int, Ge(0)] = 1
    """Number of Megabytes of data to transmit"""

    mtu: Annotated[int, Gt(0)] = 400
    """Size of IP packets to send in bytes"""

    duration: Annotated[float, Gt(0)] = 100.0
    """Time to allow flows to run in seconds"""

    flow_monitor: bool = False
    """Enable flow monitor"""

    queue_disc_type: str = "ns3::PfifoFastQueueDisc"
    """Queue disc type for gateway (e.g. ns3::CoDelQueueDisc)"""

    sack: bool = True
    """Enable or disable SACK option"""

    recovery: str = "ns3::TcpClassicRecovery"
    """Recovery algorithm type to use (e.g. ns3::TcpPrrRecovery"""

    transport_prot: str = "TcpRlTimeBased"
    """Transport protocol to use: TcpNewReno, TcpHybla, TcpHighSpeed, TcpHtcp, 
    TcpVegas, TcpScalable, TcpVeno, TcpBic, TcpYeah, TcpIllinois, TcpWestwood, 
    TcpWestwoodPlus, TcpLedbat, TcpLp, TcpRlTimeBased, TcpRlEventBased"""

    def asdict(self) -> dict:
        """Return dataclass as dict

        Returns:
            dict: dataclass as dict
        """
        return {k: v for k, v in asdict(self).items()}
